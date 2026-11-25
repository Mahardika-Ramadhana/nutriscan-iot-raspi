"""Environment, dependency, model, and config readiness checker for NutriScan."""

from __future__ import annotations

import importlib
from pathlib import Path
import platform
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_module(name: str, label: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"[OK]   Python package: {label}")
        return True
    except Exception as exc:
        print(f"[MISS] Python package: {label} ({exc})")
        return False


def main() -> int:
    print(f"=== NutriScan Environment Check ({platform.system()} / {platform.machine()}) ===")
    print(f"Python: {sys.version.split()[0]}")

    core_ok = True
    for mod, label in (
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("requests", "requests"),
    ):
        core_ok = _check_module(mod, label) and core_ok

    ncnn_ok = _check_module("ncnn", "ncnn (Native NCNN runtime)")
    if not ncnn_ok:
        _check_module("ultralytics", "ultralytics (fallback opsional)")

    for mod, label in (
        ("cv2", "opencv / cv2"),
        ("tkinter", "tkinter (Live preview GUI)"),
        ("picamera2", "picamera2 (Pi Camera CSI)"),
        ("adafruit_hx711", "adafruit-circuitpython-hx711"),
        ("lgpio", "lgpio (Bookworm GPIO)"),
        ("gpiozero", "gpiozero (Tombol fisik)"),
        ("flask", "Flask (Web Dashboard)"),
        ("paho.mqtt.client", "paho-mqtt"),
        ("supabase", "supabase"),
    ):
        _check_module(mod, label)

    print("\n=== Aset Model & Data ===")
    required_paths = [
        _PROJECT_ROOT / "models" / "best_640_ncnn_model" / "model.ncnn.bin",
        _PROJECT_ROOT / "models" / "best_640_ncnn_model" / "model.ncnn.param",
        _PROJECT_ROOT / "models" / "best_416_ncnn_model" / "model.ncnn.bin",
        _PROJECT_ROOT / "models" / "best_416_ncnn_model" / "model.ncnn.param",
        _PROJECT_ROOT / "data" / "classes_nutritions.csv",
        _PROJECT_ROOT / "data" / "config.example.json",
    ]
    assets_ok = True
    for p in required_paths:
        exists = p.is_file()
        print(f"[{'OK' if exists else 'MISS'}] {p.relative_to(_PROJECT_ROOT)}")
        assets_ok = assets_ok and exists

    prod_cfg = _PROJECT_ROOT / "data" / "config.json"
    print(
        f"[{'OK' if prod_cfg.is_file() else 'WARN'}] data/config.json "
        f"({'ditemukan' if prod_cfg.is_file() else 'belum dibuat; salin dari data/config.example.json'})"
    )
    return 0 if (core_ok and assets_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
