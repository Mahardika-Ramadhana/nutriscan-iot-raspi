from dataclasses import replace
from pathlib import Path
import pytest

from nutriscan.core.app import NutriScanApp, format_analysis_report, main as app_main, parse_args
from nutriscan.core.config import load_config, with_mock_overrides
from nutriscan.core.logger import configure_logging, get_logger
from nutriscan.core.state_machine import ScanState


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_end_to_end_pipeline_mock_once_cycle(tmp_path: Path):
    base = load_config(_repo_root() / "data" / "config.example.json")
    cfg = with_mock_overrides(
        replace(
            base,
            storage=replace(
                base.storage,
                db_path=tmp_path / "nutriscan.db",
                capture_dir=tmp_path / "captures",
                save_annotated_image=True,
            ),
            loadcell=replace(
                base.loadcell,
                sample_interval_s=0.2,
                mock=replace(
                    base.loadcell.mock,
                    enabled=True,
                    weights_g=(0.0, 350.0, 350.2, 0.0),
                    static_weight_g=None,
                    noise_g=0.2,
                    hold_seconds_per_step=2.0,
                ),
            ),
        )
    )
    clock = FakeClock()
    app = NutriScanApp(cfg, clock=clock, sleeper=clock.sleep)
    app.initialize()

    try:
        payload = app.run_once_sync(max_steps=100)
        assert payload is not None
        assert app.sm.state == ScanState.IDLE
        assert payload["berat_total_g"] == pytest.approx(350.0, abs=1.5)
        assert set(payload["komponen"].keys()) == {"Nasi", "Ayam Goreng", "Sawi"}
        assert payload["total_gizi"]["energi_kcal"] > 400.0

        scans = app.db.get_all_scans()
        assert len(scans) == 1
        assert scans[0]["sync_status"] == "local_only"
        assert len(list(cfg.storage.capture_dir.glob("*_hasil.jpg"))) == 1
    finally:
        app.stop()


def test_app_step_transitions_manual_trigger_and_logging(tmp_path: Path):
    log_dir = tmp_path / "logs"
    configure_logging("DEBUG", log_dir=log_dir, enable_file=True)
    log = get_logger("PipelineTest")
    log.info("logger check")
    assert len(list(log_dir.glob("nutriscan_*.log"))) == 1

    base = load_config(_repo_root() / "data" / "config.example.json")
    cfg = with_mock_overrides(
        replace(
            base,
            storage=replace(base.storage, db_path=tmp_path / "app.db", capture_dir=tmp_path / "cap"),
        )
    )
    clock = FakeClock(10.0)
    app = NutriScanApp(cfg, clock=clock, sleeper=clock.sleep)
    app.initialize()
    try:
        # Abort stabilizing if weight drops before stability_duration_s
        assert app.step_sensor(100.0, clock()) == ScanState.STABILIZING
        assert app.step_sensor(5.0, clock.sleep(0.5) or clock()) == ScanState.IDLE

        # Tare scale & manual trigger
        app.tare_scale()
        assert app.trigger_manual_scan() is True
        res = app.process_scan(locked_weight_g=250.0)
        assert res is not None
        assert "Nasi" in format_analysis_report(app.detector.analyze(app.last_raw_frame_bgr, 250.0)[0])

        # CLI arg parser check
        parsed = parse_args(["--mock", "--once", "--no-web", "--log-level", "WARNING"])
        assert parsed.mock and parsed.once and parsed.no_web and parsed.log_level == "WARNING"
    finally:
        app.stop()
        # Reset root file handlers so tmp_path cleans cleanly
        configure_logging("INFO", enable_file=False)


def test_cli_main_mock_once_with_temp_config(tmp_path: Path):
    import json

    raw = json.loads((_repo_root() / "data" / "config.example.json").read_text(encoding="utf-8"))
    raw["storage"]["db_path"] = str(tmp_path / "cli.db")
    raw["storage"]["capture_dir"] = str(tmp_path / "cli_cap")
    raw["loadcell"]["sample_interval_s"] = 0.2
    raw["loadcell"]["mock"]["hold_seconds_per_step"] = 0.6
    raw["loadcell"]["mock"]["weights_g"] = [0.0, 320.0, 320.2, 320.1, 320.0, 0.0, 0.0]
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")

    rc = app_main(["--config", str(cfg_path), "--mock", "--once", "--no-web"])
    assert rc == 0
    assert (tmp_path / "cli.db").is_file()


def test_app_error_recovery_and_threaded_start_stop(tmp_path: Path):
    base = load_config(_repo_root() / "data" / "config.example.json")
    cfg = with_mock_overrides(
        replace(
            base,
            storage=replace(base.storage, db_path=tmp_path / "th.db", capture_dir=tmp_path / "th_cap"),
            loadcell=replace(base.loadcell, sample_interval_s=0.05),
        )
    )
    app = NutriScanApp(cfg)
    app.start(enable_web=False)
    try:
        # Force an error inside process_scan -> must recover cleanly to COOLDOWN
        app.sm.trigger_processing(100.0)
        res = app.process_scan(locked_weight_g=100.0, frame=app.camera.capture_frame()[:0, :0, :])
        assert res is None
        assert app.sm.state == ScanState.COOLDOWN
    finally:
        app.stop()


