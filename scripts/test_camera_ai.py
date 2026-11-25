"""Quick camera + YOLO26n-seg NCNN test runner without Load Cell HX711/GPIO."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nutriscan.ai.detector import NutriScanDetector  # noqa: E402
from nutriscan.core.app import format_analysis_report  # noqa: E402
from nutriscan.core.config import AppConfig, load_config  # noqa: E402
from nutriscan.core.logger import configure_logging  # noqa: E402
from nutriscan.db.cloud_sync import save_frame_image  # noqa: E402
from nutriscan.hardware.camera import CameraManager  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tes langsung Kamera + Model AI NCNN NutriScan tanpa Load Cell"
    )
    parser.add_argument("--config", default=None, help="Path file JSON konfigurasi")
    parser.add_argument(
        "--backend",
        choices=("auto", "picamera2", "opencv"),
        default=None,
        help="Paksa backend kamera tertentu (default: sesuai config)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="Index USB webcam untuk OpenCV (misal 0 atau 2)",
    )
    parser.add_argument(
        "--variant",
        choices=("640", "416"),
        default=None,
        help="Pilih varian model NCNN (640 atau 416)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=350.0,
        help="Berat simulasi dalam gram untuk kalkulasi gizi (default: 350.0)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Mode live preview kamera + deteksi AI berulang (tekan 'q' atau Ctrl+C untuk keluar)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Jangan buka jendela preview OpenCV pada mode --live (cetak ke terminal saja)",
    )
    parser.add_argument(
        "--output",
        default="data/captures/test_camera_ai.jpg",
        help="Lokasi simpan gambar hasil segmentasi (mode 1-shot)",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    cam_cfg = cfg.camera
    if args.backend is not None:
        cam_cfg = replace(cam_cfg, backend=args.backend)
    if args.camera_id is not None:
        cam_cfg = replace(cam_cfg, camera_ids=(args.camera_id,))

    model_cfg = replace(cfg.model, mock_mode=False)
    if args.variant is not None:
        imgsz = int(args.variant)
        model_cfg = replace(
            model_cfg,
            variant=args.variant,
            imgsz=imgsz,
            path=(cfg.project_root / "models" / f"best_{args.variant}_ncnn_model").resolve(),
        )

    return replace(cfg, camera=cam_cfg, model=model_cfg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="INFO", enable_file=False)

    cfg = _apply_cli_overrides(load_config(args.config, allow_example_fallback=True), args)
    detector = NutriScanDetector(cfg.model)
    detector.load_model()

    with CameraManager(cfg.camera) as cam:
        print(f"Kamera aktif: {cam.active_backend_name}")
        if not args.live:
            frame = cam.capture_frame()
            analysis, annotated = detector.analyze(frame, weight_total_g=args.weight)
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = cfg.project_root / out_path
            saved = save_frame_image(
                annotated if annotated is not None else frame,
                out_path,
            )
            print("\n" + format_analysis_report(analysis))
            print(f"\nGambar hasil segmentasi disimpan di: {saved}")
            return 0

        cv2_mod = None
        if not args.no_gui:
            try:
                import cv2

                cv2_mod = cv2
            except ImportError:
                cv2_mod = None

        print("Mode --live aktif. Tekan 'q' di jendela kamera atau Ctrl+C di terminal untuk berhenti.")
        try:
            while True:
                t0 = time.perf_counter()
                frame = cam.capture_frame()
                analysis, annotated = detector.analyze(frame, weight_total_g=args.weight)
                fps = 1.0 / max(1e-6, time.perf_counter() - t0)
                print(f"\n[FPS: {fps:.2f}] Deteksi: {analysis.detection_count}")
                print(format_analysis_report(analysis))

                if cv2_mod is not None:
                    display = annotated if annotated is not None else frame
                    cv2_mod.imshow("NutriScan MBG - Live Camera AI Test (tekan q keluar)", display)
                    if (cv2_mod.waitKey(1) & 0xFF) == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\nDihentikan oleh pengguna.")
        finally:
            if cv2_mod is not None:
                cv2_mod.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
