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

from nutriscan.core.runtime_env import (  # noqa: E402
    apply_arm_runtime_guards,
    configure_opencv_threads,
    reexec_with_arm_guards_if_needed,
)

if __name__ == "__main__":
    reexec_with_arm_guards_if_needed(
        enable_gui=("--live" in sys.argv and "--no-gui" not in sys.argv and "--gui=none" not in sys.argv)
    )

import numpy as np  # noqa: E402

from nutriscan.ai.detector import NutriScanDetector  # noqa: E402
from nutriscan.core.app import format_analysis_report  # noqa: E402
from nutriscan.core.config import AppConfig, load_config  # noqa: E402
from nutriscan.core.logger import configure_logging  # noqa: E402
from nutriscan.db.cloud_sync import save_frame_image  # noqa: E402
from nutriscan.hardware.camera import CameraManager  # noqa: E402


class TkPreviewWindow:
    """Zero-Qt live preview window using Python stdlib tkinter + binary PPM."""

    def __init__(self, title: str = "NutriScan MBG - Live Camera AI Test (tekan q keluar)") -> None:
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title(title)
        self._closed = False
        self._photo = None
        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill="both", expand=True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<KeyPress-q>", lambda _e: self._on_close())
        self._root.bind("<KeyPress-Q>", lambda _e: self._on_close())
        self._root.bind("<Escape>", lambda _e: self._on_close())

    def _on_close(self) -> None:
        self._closed = True

    def show(self, frame_bgr: np.ndarray) -> bool:
        if self._closed:
            return False
        rgb = frame_bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        if w > 960:
            step = max(1, (w + 959) // 960)
            rgb = rgb[::step, ::step]
            h, w = rgb.shape[:2]
        ppm_data = f"P6 {w} {h} 255\n".encode("ascii") + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        photo = self._tk.PhotoImage(data=ppm_data, format="PPM")
        self._photo = photo
        self._label.configure(image=photo)
        self._root.update_idletasks()
        self._root.update()
        return not self._closed

    def close(self) -> None:
        self._closed = True
        try:
            self._root.destroy()
        except Exception:
            pass


class Cv2PreviewWindow:
    """Fallback OpenCV HighGUI preview window with single-thread guard."""

    def __init__(self, title: str = "NutriScan MBG - Live Camera AI Test (tekan q keluar)") -> None:
        configure_opencv_threads()
        import cv2

        self._cv2 = cv2
        self._title = title

    def show(self, frame_bgr: np.ndarray) -> bool:
        self._cv2.imshow(self._title, frame_bgr)
        return (self._cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        try:
            self._cv2.destroyAllWindows()
        except Exception:
            pass


def _create_preview_window(gui_mode: str):
    if gui_mode == "none":
        return None
    if gui_mode in ("auto", "tk"):
        try:
            return TkPreviewWindow()
        except Exception as exc:
            if gui_mode == "tk":
                print(f"[INFO] Preview Tkinter tidak tersedia ({exc}); lanjut mode terminal.")
                return None
    if gui_mode in ("auto", "cv2"):
        try:
            return Cv2PreviewWindow()
        except Exception as exc:
            print(f"[INFO] Preview OpenCV tidak tersedia ({exc}); lanjut mode terminal.")
    return None


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
        "--gui",
        choices=("auto", "tk", "cv2", "none"),
        default="auto",
        help="Backend jendela preview pada mode --live (default: auto, prioritas Tkinter bebas Qt5)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Jangan buka jendela preview pada mode --live (cetak ke terminal saja)",
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
    gui_mode = "none" if args.no_gui else args.gui
    apply_arm_runtime_guards(enable_gui=(args.live and gui_mode != "none"))
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
            saved = save_frame_image(annotated if annotated is not None else frame, out_path)
            print("\n" + format_analysis_report(analysis))
            print(f"\nGambar hasil segmentasi disimpan di: {saved}")
            return 0

        preview = _create_preview_window(gui_mode)
        print("Mode --live aktif. Tekan 'q' di jendela kamera atau Ctrl+C di terminal untuk berhenti.")
        try:
            while True:
                t0 = time.perf_counter()
                frame = cam.capture_frame()
                analysis, annotated = detector.analyze(frame, weight_total_g=args.weight)
                fps = 1.0 / max(1e-6, time.perf_counter() - t0)
                print(f"\n[FPS: {fps:.2f}] Deteksi: {analysis.detection_count}")
                print(format_analysis_report(analysis))

                if preview is not None:
                    display = annotated if annotated is not None else frame
                    if not preview.show(display):
                        break
        except KeyboardInterrupt:
            print("\nDihentikan oleh pengguna.")
        finally:
            if preview is not None:
                preview.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
