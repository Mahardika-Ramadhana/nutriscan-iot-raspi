import json
from pathlib import Path
import pytest

from nutriscan.core.config import (
    ConfigError,
    get_secret,
    load_config,
    load_dotenv_file,
    with_mock_overrides,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_config_example_success():
    cfg = load_config(_repo_root() / "data" / "config.example.json")
    assert cfg.device.id == "NUTRISCAN-PI-01"
    assert cfg.model.variant == "640"
    assert cfg.model.imgsz == 640
    assert cfg.model.path == (_repo_root() / "models" / "best_640_ncnn_model").resolve()
    assert cfg.hardware.pin_loadcell_dt == 5
    assert cfg.hardware.pin_loadcell_sck == 6
    assert cfg.hardware.pin_tombol == 17


def test_load_config_missing_without_fallback_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="tidak ditemukan"):
        load_config(tmp_path / "missing.json", allow_example_fallback=False)


def test_load_config_fallback_to_example(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    example = _repo_root() / "data" / "config.example.json"
    (data_dir / "config.example.json").write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = load_config(data_dir / "config.json", allow_example_fallback=True)
    assert cfg.config_path.name == "config.example.json"
    assert cfg.project_root == tmp_path.resolve()


def test_load_config_invalid_json_raises(tmp_path: Path):
    bad = tmp_path / "config.json"
    bad.write_text("{broken json", encoding="utf-8")
    with pytest.raises(ConfigError, match="rusak"):
        load_config(bad)


def test_load_config_rejects_duplicate_gpio_pins(tmp_path: Path):
    raw = json.loads((_repo_root() / "data" / "config.example.json").read_text(encoding="utf-8"))
    raw["hardware"]["pin_loadcell_sck"] = raw["hardware"]["pin_loadcell_dt"]
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="tidak boleh sama"):
        load_config(cfg_file)


def test_load_config_rejects_invalid_thresholds(tmp_path: Path):
    raw = json.loads((_repo_root() / "data" / "config.example.json").read_text(encoding="utf-8"))
    raw["loadcell"]["removal_threshold_g"] = raw["loadcell"]["trigger_threshold_g"]
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="removal_threshold_g"):
        load_config(cfg_file)


def test_with_mock_overrides_returns_new_immutable_copy():
    original = load_config(_repo_root() / "data" / "config.example.json")
    mocked = with_mock_overrides(original)

    assert original.environment == "development"
    assert original.model.mock_mode is False
    assert mocked.environment == "mock"
    assert mocked.model.mock_mode is True
    assert mocked.camera.backend == "mock"
    assert mocked.loadcell.backend == "mock"
    assert mocked.loadcell.mock.enabled is True
    assert mocked.hardware.enable_physical_button is False


def test_load_dotenv_and_get_secret(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NUTRISCAN_TEST_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nNUTRISCAN_TEST_TOKEN='secret-123'\n", encoding="utf-8")

    load_dotenv_file(env_file)
    assert get_secret("NUTRISCAN_TEST_TOKEN") == "secret-123"
    assert get_secret("MISSING_ENV_VAR", "fallback-val") == "fallback-val"
