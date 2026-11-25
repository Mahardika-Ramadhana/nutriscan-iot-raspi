"""Main NutriScan application orchestrator (DICA Core-Pinning + State Machine)."""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

from nutriscan.ai.detector import NutriScanDetector, ScanAnalysis
from nutriscan.api.web_server import start_web_server
from nutriscan.core.config import (
    AppConfig,
    load_config,
    with_camera_only_overrides,
    with_mock_overrides,
)
from nutriscan.core.logger import configure_logging, get_logger
from nutriscan.core.state_machine import NutriScanStateMachine, ScanState
from nutriscan.db.cloud_sync import CloudSync
from nutriscan.db.database import Database
from nutriscan.hardware.camera import CameraManager
from nutriscan.hardware.loadcell import LoadCell

logger = get_logger("NutriScanApp")


def _try_pin_core(core_id: int, worker_name: str) -> None:
    """Pin current thread to a specific Raspberry Pi 4 CPU core when supported."""
    try:
        os.sched_setaffinity(0, {core_id})
        logger.debug("%s dipin ke CPU Core %d", worker_name, core_id)
    except (AttributeError, OSError):
        pass


def format_analysis_report(analysis: ScanAnalysis) -> str:
    """Format human-readable report matching yolo/detect.py output style."""
    w = analysis.timing_ms
    lines = [
        f"Waktu AI: {w.get('total', 0):.0f} ms "
        f"(pre {w.get('preprocess', 0):.0f} / infer {w.get('inference', 0):.0f} / post {w.get('postprocess', 0):.0f})"
    ]
    if not analysis.components:
        lines.append("Tidak ada makanan terdeteksi.")
        return "\n".join(lines)

    for name, comp in sorted(analysis.components.items(), key=lambda item: -item[1].proportion):
        row = f"  - {name:<22} {comp.proportion * 100:5.1f}%"
        if comp.estimated_grams is not None:
            row += f"  ~{comp.estimated_grams:.1f} g"
        if comp.nutrition and comp.nutrition.get("energi_kcal") is not None:
            row += f"  ~{comp.nutrition['energi_kcal']:.1f} kkal"
        lines.append(row)

    if analysis.total_nutrition:
        summary = ", ".join(f"{k}={v}" for k, v in analysis.total_nutrition.items())
        lines.append(f"Total gizi estimasi: {summary}")
    return "\n".join(lines)


class NutriScanApp:
    """Headless multi-threaded Edge AI + Load Cell coordinator."""

    def __init__(
        self,
        config: AppConfig,
        loadcell: LoadCell | None = None,
        camera: CameraManager | None = None,
        detector: NutriScanDetector | None = None,
        db: Database | None = None,
        cloud_sync: CloudSync | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._clock = clock
        self._sleeper = sleeper

        self.db = db or Database(config.storage.db_path)
        self.loadcell = loadcell or LoadCell(
            config.loadcell,
            config.hardware,
            environment=config.environment,
            clock=clock,
            sleeper=sleeper,
        )
        self.camera = camera or CameraManager(
            config.camera,
            environment=config.environment,
            sleeper=sleeper,
        )
        self.detector = detector or NutriScanDetector(config.model)
        self.cloud_sync = cloud_sync or CloudSync(config, self.db)
        self.sm = NutriScanStateMachine()

        self.last_raw_frame_bgr: np.ndarray | None = None
        self.last_drawn_frame_bgr: np.ndarray | None = None

        self._btn_fisik = None
        self._running = False
        self._snapshot_event = threading.Event()
        self._frame_queue: queue.Queue[tuple[np.ndarray, float | None]] = queue.Queue(maxsize=1)
        self._threads: list[threading.Thread] = []

        self._cooldown_started_at: float = 0.0
        self._last_flush_at: float = clock()

    def initialize(self) -> None:
        """Initialize database, camera, load cell (auto-tare), AI model, and optional button."""
        logger.info("Menginisialisasi sistem NutriScan MBG (env=%s)...", self.config.environment)
        self.db.init_db()
        self.camera.init_camera()
        self.loadcell.init_loadcell()
        self.detector.load_model()

        if self.config.hardware.enable_physical_button:
            try:
                from gpiozero import Button

                self._btn_fisik = Button(self.config.hardware.pin_tombol, pull_up=True)
                self._btn_fisik.when_pressed = self.trigger_manual_scan
                logger.info("Tombol fisik GPIO%d aktif", self.config.hardware.pin_tombol)
            except Exception as exc:
                logger.warning("Tombol fisik GPIO tidak aktif: %s", exc)

        if self.cloud_sync.has_remote_transports():
            self.cloud_sync.sync_pending(limit=10)

    def tare_scale(self) -> float:
        offset = self.loadcell.tare()
        self.sm.update_weight(0.0, 0.0)
        return offset

    def trigger_manual_scan(self) -> bool:
        """Manually trigger capture/inference via button or Web API."""
        weight = self.sm.current_weight_g
        if self.sm.trigger_processing(weight):
            self._snapshot_event.set()
            return True
        return False

    def step_sensor(self, weight_g: float, now: float) -> ScanState:
        """Advance state machine based on current weight sample and monotonic timestamp."""
        state = self.sm.state
        lc_cfg = self.config.loadcell

        if state == ScanState.IDLE:
            self.sm.update_weight(weight_g, 0.0)
            if (
                self.cloud_sync.has_remote_transports()
                and (now - self._last_flush_at) >= self.config.sync.flush_interval_s
            ):
                self.cloud_sync.sync_pending(limit=10)
                self._last_flush_at = now

            if weight_g >= lc_cfg.trigger_threshold_g:
                self.sm.begin_stabilizing(weight_g)
                self.loadcell.stability.evaluate(weight_g, now)
            return self.sm.state

        if state == ScanState.STABILIZING:
            status = self.loadcell.stability.evaluate(weight_g, now)
            self.sm.update_weight(status.weight_g, status.stable_elapsed_s)
            if not status.is_above_trigger:
                self.sm.abort_stabilizing()
            elif status.is_stable and self.sm.trigger_processing(status.weight_g):
                self._snapshot_event.set()
            return self.sm.state

        if state == ScanState.COOLDOWN:
            self.sm.update_weight(weight_g, 0.0)
            cooldown_elapsed = (now - self._cooldown_started_at) >= lc_cfg.cooldown_min_s
            if cooldown_elapsed and self.loadcell.stability.is_plate_removed(weight_g, now):
                self.loadcell.stability.reset()
                self.sm.complete_cooldown()
            return self.sm.state

        self.sm.update_weight(weight_g, 0.0)
        return state

    def process_scan(
        self,
        locked_weight_g: float,
        frame: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Run capture (if needed), NCNN segmentation, nutrition split, and DB/cloud sync."""
        try:
            captured = frame if frame is not None else self.camera.capture_frame()
            self.last_raw_frame_bgr = captured.copy()

            analysis, annotated = self.detector.analyze(captured, weight_total_g=locked_weight_g)
            self.last_drawn_frame_bgr = annotated if annotated is not None else captured.copy()
            logger.info("\n%s", format_analysis_report(analysis))

            if analysis.detection_count == 0 and not self.config.storage.send_empty_detections:
                self.sm.skip_empty_processing()
                self._enter_cooldown()
                return None

            temp_payload = analysis.to_dict()
            self.sm.finish_processing(temp_payload, has_occlusion=analysis.has_occlusion)
            saved_payload = self.cloud_sync.record_and_sync(
                analysis,
                raw_frame=self.last_raw_frame_bgr,
                annotated_frame=self.last_drawn_frame_bgr,
            )
            # Perbarui snapshot terakhir dengan scan_id & timestamp_utc final
            self.sm._last_scan_payload = dict(saved_payload)  # noqa: SLF001
            self.sm.finish_sending()
            self._enter_cooldown()
            return saved_payload
        except Exception as exc:
            logger.error("Error pipeline deteksi/pengiriman: %s", exc, exc_info=True)
            self.sm.cancel_processing(str(exc))
            self._enter_cooldown()
            return None

    def _enter_cooldown(self) -> None:
        self._cooldown_started_at = self._clock()
        self.loadcell.stability.reset()

    def _camera_task(self) -> None:
        _try_pin_core(1, "CamThread")
        while self._running:
            if not self._snapshot_event.wait(timeout=0.1):
                continue
            self._snapshot_event.clear()
            snap = self.sm.snapshot()
            try:
                frame = self.camera.capture_frame()
                self.last_raw_frame_bgr = frame.copy()
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self._frame_queue.put((frame, snap.locked_weight_g))
            except Exception as exc:
                logger.error("Gagal mengambil snapshot kamera: %s", exc)
                self.sm.cancel_processing(str(exc))
                self._enter_cooldown()

    def _ai_task(self) -> None:
        _try_pin_core(3, "AIThread")
        while self._running:
            try:
                frame, locked_weight_g = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.process_scan(
                locked_weight_g=locked_weight_g if locked_weight_g is not None else 0.0,
                frame=frame,
            )

    def _sensor_task(self) -> None:
        _try_pin_core(2, "SensorThread")
        while self._running:
            try:
                sample = self.loadcell.read_sample()
                self.step_sensor(sample.weight_g, sample.timestamp)
            except Exception as exc:
                logger.error("Error loop sensor LoadCell: %s", exc)
                self._sleeper(0.5)
            self._sleeper(self.config.loadcell.sample_interval_s)

    def run_once_sync(self, max_steps: int = 300) -> dict[str, Any] | None:
        """Run synchronously until one full scan + cooldown cycle finishes (for --once & tests)."""
        completed_payload: dict[str, Any] | None = None
        scan_done = False

        for _ in range(max_steps):
            sample = self.loadcell.read_sample()
            state = self.step_sensor(sample.weight_g, sample.timestamp)
            if state == ScanState.PROCESSING:
                snap = self.sm.snapshot()
                completed_payload = self.process_scan(
                    locked_weight_g=snap.locked_weight_g or sample.weight_g,
                )
                scan_done = True
            elif scan_done and state == ScanState.IDLE:
                return completed_payload
            self._sleeper(self.config.loadcell.sample_interval_s)

        return completed_payload

    def start(self, enable_web: bool = True) -> None:
        """Start multi-threaded production workers (DICA Core-Pinning)."""
        self.initialize()
        _try_pin_core(0, "MainThread")
        self._running = True

        self._threads = [
            threading.Thread(target=self._camera_task, name="CamThread", daemon=True),
            threading.Thread(target=self._ai_task, name="AIThread", daemon=True),
            threading.Thread(target=self._sensor_task, name="SensorThread", daemon=True),
        ]
        if enable_web and self.config.web.enabled:
            self._threads.append(
                threading.Thread(
                    target=start_web_server,
                    args=(self,),
                    name="WebThread",
                    daemon=True,
                )
            )
        for thread in self._threads:
            thread.start()
        logger.info("NutriScan aktif (%d worker thread berjalan)", len(self._threads))

    def stop(self) -> None:
        """Gracefully stop background workers and release hardware."""
        logger.info("Mematikan NutriScan...")
        self._running = False
        self._snapshot_event.set()
        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()

        if self._btn_fisik is not None:
            try:
                self._btn_fisik.close()
            except Exception:
                pass
            self._btn_fisik = None
        self.camera.release_all()
        self.loadcell.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NutriScan MBG - Raspberry Pi 4 IoT Runtime")
    parser.add_argument(
        "--config",
        default=None,
        help="Path ke file JSON konfigurasi (default: data/config.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Jalankan simulasi kamera, HX711, dan model tanpa hardware fisik",
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Jalankan kamera asli + model AI NCNN asli tanpa Load Cell HX711/GPIO",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Jalankan 1 siklus scan lengkap lalu keluar",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Nonaktifkan Flask Web Dashboard lokal",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override level log (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, allow_example_fallback=(args.mock or args.camera_only))
    if args.mock:
        cfg = with_mock_overrides(cfg)
    elif args.camera_only:
        cfg = with_camera_only_overrides(cfg)

    log_level = args.log_level or cfg.device.log_level
    configure_logging(
        level=log_level,
        log_dir=cfg.project_root / "logs",
        enable_file=not args.mock,
    )

    app = NutriScanApp(cfg)
    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("Sinyal %s diterima, berhenti dengan aman...", signum)
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        if args.once:
            app.initialize()
            if args.camera_only:
                weight = cfg.loadcell.mock.static_weight_g or 350.0
                app.sm.trigger_processing(weight)
                result = app.process_scan(locked_weight_g=weight)
            else:
                result = app.run_once_sync()
            logger.info("Selesai (--once). Scan ID: %s", (result or {}).get("scan_id"))
            return 0

        app.start(enable_web=not args.no_web)
        while not stop_event.wait(timeout=0.5):
            pass
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
