import json
from pathlib import Path
import pytest

from nutriscan.core.config import load_config, with_mock_overrides
from scripts.calibrate_hx711 import (
    compute_reference_unit,
    run_interactive_calibration,
    save_calibration_to_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_compute_reference_unit_and_validation():
    assert compute_reference_unit(1000.0, 211000.0, 500.0) == pytest.approx(420.0)

    with pytest.raises(ValueError, match="> 0"):
        compute_reference_unit(0.0, 1000.0, 0.0)

    with pytest.raises(ValueError, match="terlalu kecil"):
        compute_reference_unit(1000.0, 1000.0, 500.0)


def test_save_calibration_to_config(tmp_path: Path):
    example_src = _repo_root() / "data" / "config.example.json"
    example_copy = tmp_path / "config.example.json"
    example_copy.write_text(example_src.read_text(encoding="utf-8"), encoding="utf-8")

    saved = save_calibration_to_config(example_copy, reference_unit=412.5, tare_offset=12345.67)
    assert saved.name == "config.json"

    parsed = json.loads(saved.read_text(encoding="utf-8"))
    assert parsed["loadcell"]["reference_unit"] == 412.5
    assert parsed["loadcell"]["tare_offset"] == 12345.67
    assert parsed["device"]["id"] == "NUTRISCAN-PI-01"


def test_run_interactive_calibration_mock_flow(tmp_path: Path):
    example_src = _repo_root() / "data" / "config.example.json"
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(example_src.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = with_mock_overrides(load_config(cfg_file))
    answers = iter(["", "500", "", "y"])
    logs: list[str] = []

    ref_unit, offset = run_interactive_calibration(
        cfg,
        samples=5,
        no_save=False,
        input_fn=lambda _prompt: next(answers),
        output_fn=logs.append,
    )
    assert ref_unit == pytest.approx(cfg.loadcell.reference_unit, rel=0.05)
    assert any("Kalibrasi berhasil disimpan" in line for line in logs)
    assert isinstance(offset, float)
