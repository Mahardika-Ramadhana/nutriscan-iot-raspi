"""Live HX711 Load Cell diagnostic and weight streaming CLI for NutriScan."""

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
    reexec_with_arm_guards_if_needed,
)

if __name__ == "__main__":
    reexec_with_arm_guards_if_needed(enable_gui=False)

apply_arm_runtime_guards(enable_gui=False)

from nutriscan.core.config import AppConfig, load_config, with_mock_overrides  # noqa: E402
from nutriscan.core.logger import configure_logging  # noqa: E402
from nutriscan.hardware.loadcell import LoadCell, LoadCellError  # noqa: E402


def _print_wiring_help(dt_pin: int, sck_pin: int) -> None:
    print("\n[PANDUAN WIRING HX711 -> RASPBERRY PI 4 (WAJIB 3.3V!)]")
    print("  HX711 VCC  -> Pin 1  (3.3V Power, JANGAN 5V!)")
    print("  HX711 GND  -> Pin 6  (Ground)")
    print(f"  HX711 DT   -> Pin 29 (GPIO {dt_pin})")
    print(f"  HX711 SCK  -> Pin 31 (GPIO {sck_pin})")
    print("  Load Cell  -> Merah=E+, Hitam=E-, Putih=A-, Hijau=A+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tes pembacaan langsung sensor berat Load Cell HX711")
    parser.add_argument("--config", default=None, help="Path file JSON konfigurasi")
    parser.add_argument(
        "--backend",
        choices=("auto", "lgpio", "adafruit_hx711", "mock"),
        default=None,
        help="Pilih backend HX711 (default: sesuai config)",
    )
    parser.add_argument("--dt", type=int, default=None, help="Override pin BCM DT/DOUT (default: 5)")
    parser.add_argument("--sck", type=int, default=None, help="Override pin BCM SCK (default: 6)")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Jumlah pembacaan sebelum keluar (0 = jalan terus sampai Ctrl+C)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Jeda antar pembacaan dalam detik (default: 0.25)",
    )
    parser.add_argument(
        "--no-tare",
        action="store_true",
        help="Lewati auto-tare awal untuk melihat tare_offset bawaan config",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Jalankan simulasi load cell tanpa hardware fisik",
    )
    return parser.parse_args(argv)


def _build_runtime_config(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config, allow_example_fallback=True)
    if args.mock:
        cfg = with_mock_overrides(cfg)

    hw_cfg = cfg.hardware
    if args.dt is not None:
        hw_cfg = replace(hw_cfg, pin_loadcell_dt=args.dt)
    if args.sck is not None:
        hw_cfg = replace(hw_cfg, pin_loadcell_sck=args.sck)

    lc_cfg = cfg.loadcell
    if args.backend is not None:
        lc_cfg = replace(lc_cfg, backend=args.backend)
    if args.no_tare:
        lc_cfg = replace(lc_cfg, auto_tare_on_boot=False)

    env = "mock" if (args.mock or lc_cfg.backend == "mock") else "production"
    return replace(cfg, environment=env, hardware=hw_cfg, loadcell=lc_cfg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="INFO", enable_file=False)
    cfg = _build_runtime_config(args)

    scale = LoadCell(cfg.loadcell, cfg.hardware, environment=cfg.environment)
    try:
        scale.init_loadcell()
    except LoadCellError as exc:
        print(f"\n[ERROR] Gagal inisialisasi HX711: {exc}")
        _print_wiring_help(cfg.hardware.pin_loadcell_dt, cfg.hardware.pin_loadcell_sck)
        return 1

    print(
        f"\nHX711 siap! Pin DT=GPIO{cfg.hardware.pin_loadcell_dt}, "
        f"SCK=GPIO{cfg.hardware.pin_loadcell_sck}, "
        f"ref_unit={scale.scale:.4f}, tare_offset={scale.offset:.2f}"
    )
    print("Tekan Ctrl+C untuk berhenti.\n")

    idx = 0
    try:
        while args.count <= 0 or idx < args.count:
            idx += 1
            sample = scale.read_sample()
            status = scale.stability.evaluate(sample.weight_g, sample.timestamp)
            if status.is_stable:
                tag = "STABIL (SIAP SCAN)"
            elif status.is_above_trigger:
                tag = f"STABILIZING {status.stable_elapsed_s:.1f}s"
            else:
                tag = "IDLE"
            print(
                f"[#{idx:03d}] Berat: {sample.weight_g:8.2f} g | "
                f"Raw ADC: {sample.raw_value:11.1f} | "
                f"Status: {tag} (spread {status.spread_g:.2f}g)"
            )
            if args.count <= 0 or idx < args.count:
                time.sleep(max(0.01, args.interval))
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna.")
    except LoadCellError as exc:
        print(f"\n[ERROR] Pembacaan HX711 terputus: {exc}")
        _print_wiring_help(cfg.hardware.pin_loadcell_dt, cfg.hardware.pin_loadcell_sck)
        return 1
    finally:
        scale.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
