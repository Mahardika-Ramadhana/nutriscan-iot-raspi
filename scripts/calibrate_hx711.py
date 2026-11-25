"""Interactive CLI calibration for HX711 Load Cell saving directly into data/config.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nutriscan.core.config import AppConfig, load_config, with_mock_overrides  # noqa: E402
from nutriscan.hardware.loadcell import LoadCell, LoadCellError  # noqa: E402


def compute_reference_unit(
    raw_empty: float,
    raw_loaded: float,
    known_weight_g: float,
) -> float:
    """Compute HX711 reference_unit = (raw_loaded - raw_empty) / known_weight_g."""
    if known_weight_g <= 0:
        raise ValueError("Berat benda referensi harus > 0 gram")
    delta = raw_loaded - raw_empty
    if abs(delta) < 1e-6:
        raise ValueError("Perbedaan nilai raw kosong & berbeban terlalu kecil; cek wiring HX711")
    return round(float(delta / known_weight_g), 4)


def save_calibration_to_config(
    config_path: Path,
    reference_unit: float,
    tare_offset: float,
) -> Path:
    """Persist calibrated reference_unit and tare_offset into data/config.json cleanly."""
    target = config_path
    if target.name == "config.example.json":
        target = target.with_name("config.json")
        if not target.exists():
            target.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    data = json.loads(target.read_text(encoding="utf-8"))
    if "loadcell" not in data or not isinstance(data["loadcell"], dict):
        raise ValueError(f"Section 'loadcell' tidak ditemukan di {target}")

    data["loadcell"]["reference_unit"] = float(reference_unit)
    data["loadcell"]["tare_offset"] = float(round(tare_offset, 2))
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def run_interactive_calibration(
    cfg: AppConfig,
    samples: int = 20,
    no_save: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    scale: LoadCell | None = None,
) -> tuple[float, float]:
    """Guide operator through empty tare, known weight sampling, verification, and save."""
    managed_scale = scale or LoadCell(cfg.loadcell, cfg.hardware, environment=cfg.environment)
    managed_scale.init_loadcell()
    try:
        output_fn("=== KALIBRASI LOAD CELL HX711 (NUTRISCAN MBG) ===")
        output_fn(
            f"Pin aktif: DT=GPIO{cfg.hardware.pin_loadcell_dt}, "
            f"SCK=GPIO{cfg.hardware.pin_loadcell_sck}, Gain={cfg.loadcell.gain}"
        )

        input_fn("\n[1/3] Kosongkan nampan/piring di atas load cell, lalu tekan ENTER...")
        raw_empty = managed_scale.read_raw_average(samples)
        managed_scale.offset = raw_empty
        output_fn(f"Offset nol (tare) tercatat: {raw_empty:.2f}")

        raw_weight_str = input_fn(
            "\n[2/3] Ketik berat benda referensi yang diketahui pasti (gram, misal 500): "
        ).strip()
        try:
            known_weight_g = float(raw_weight_str.replace(",", "."))
        except ValueError as exc:
            raise LoadCellError(f"Input berat tidak valid: '{raw_weight_str}'") from exc

        if cfg.loadcell.backend == "mock" or managed_scale.is_mock:
            managed_scale.set_mock_weight(known_weight_g)

        input_fn(f"Letakkan benda seberat {known_weight_g:.1f} g di atas timbangan, lalu tekan ENTER...")
        raw_loaded = managed_scale.read_raw_average(samples)
        ref_unit = compute_reference_unit(raw_empty, raw_loaded, known_weight_g)
        managed_scale.scale = ref_unit
        output_fn(f"Nilai raw berbeban : {raw_loaded:.2f}")
        output_fn(f"Reference unit baru: {ref_unit:.4f}")

        output_fn("\n[3/3] Verifikasi 3 pembacaan langsung:")
        for idx in range(3):
            reading = managed_scale.read_weight()
            output_fn(f"  Sampel #{idx + 1}: {reading:.1f} g")

        if not no_save:
            confirm = input_fn(
                f"\nSimpan ke {cfg.config_path.with_name('config.json')}? [Y/n]: "
            ).strip().lower()
            if confirm in ("", "y", "yes", "ya"):
                saved_path = save_calibration_to_config(cfg.config_path, ref_unit, raw_empty)
                output_fn(f"Kalibrasi berhasil disimpan ke {saved_path}")
        return ref_unit, raw_empty
    finally:
        if scale is None:
            managed_scale.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalibrasi interaktif Load Cell HX711 NutriScan")
    parser.add_argument("--config", default=None, help="Path file JSON konfigurasi")
    parser.add_argument("--samples", type=int, default=20, help="Jumlah sampel per langkah")
    parser.add_argument("--mock", action="store_true", help="Jalankan simulasi kalibrasi tanpa hardware")
    parser.add_argument("--no-save", action="store_true", help="Jangan tulis hasil ke file config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, allow_example_fallback=True)
    if args.mock:
        cfg = with_mock_overrides(cfg)
    run_interactive_calibration(cfg, samples=args.samples, no_save=args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
