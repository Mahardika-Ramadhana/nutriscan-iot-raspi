import sys
import types
import pytest

from nutriscan.core.config import HardwarePinsConfig, LoadCellConfig, LoadCellMockConfig
from nutriscan.hardware.loadcell import (
    AdafruitHX711Backend,
    BitBangHX711Backend,
    LgpioPinDriver,
    LoadCell,
    LoadCellError,
    MockHX711Backend,
    RpiGpioPinDriver,
    trimmed_mean,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class FakePinDriver:
    def __init__(self, bits_24: int, ready: bool = True) -> None:
        self._bits = [(bits_24 >> (23 - i)) & 1 for i in range(24)]
        self._idx = -1
        self._ready = ready
        self.closed = False

    def read_dout(self) -> int:
        if not self._ready:
            return 1
        if self._idx < 0:
            return 0  # Ready (DOUT LOW before first SCK pulse)
        if self._idx < len(self._bits):
            return self._bits[self._idx]
        return 0

    def write_sck(self, level: int) -> None:
        if level == 1 and self._idx < len(self._bits):
            self._idx += 1

    def close(self) -> None:
        self.closed = True


def _make_loadcell_cfg(**overrides) -> LoadCellConfig:
    mock_cfg = overrides.pop(
        "mock",
        LoadCellMockConfig(
            enabled=True,
            weights_g=(0.0, 250.0, 0.0),
            static_weight_g=None,
            noise_g=0.0,
            hold_seconds_per_step=2.0,
        ),
    )
    defaults = dict(
        backend="mock",
        gain=128,
        reference_unit=400.0,
        tare_offset=1000.0,
        auto_tare_on_boot=True,
        tare_samples=6,
        read_samples=6,
        sample_interval_s=0.1,
        deadband_g=5.0,
        trigger_threshold_g=30.0,
        removal_threshold_g=10.0,
        stability_tolerance_g=3.0,
        stability_duration_s=1.5,
        removal_confirm_s=0.8,
        cooldown_min_s=1.0,
        mock=mock_cfg,
    )
    defaults.update(overrides)
    return LoadCellConfig(**defaults)


def _pins() -> HardwarePinsConfig:
    return HardwarePinsConfig(pin_loadcell_dt=5, pin_loadcell_sck=6, pin_tombol=17, enable_physical_button=False)


def test_trimmed_mean_drops_outliers():
    # Sorted: [-999, -50, 100, 102, 104, 106, 9999] -> valid [100, 102, 104] -> avg 102.0
    result = trimmed_mean([9999.0, 100.0, -999.0, 106.0, 102.0, -50.0, 104.0])
    assert result == pytest.approx(102.0)
    assert trimmed_mean([10.0, 20.0]) == pytest.approx(15.0)
    with pytest.raises(LoadCellError, match="Tidak ada sampel"):
        trimmed_mean([])


def test_bitbang_hx711_sign_extension_and_saturation():
    # Negative 24-bit value: -1000 => 0x1000000 - 1000 = 0xFFFC18
    pin_driver = FakePinDriver(0xFFFC18)
    reader = BitBangHX711Backend(pin_driver, gain=128, max_read_retries=1, sleeper=lambda _: None)
    assert reader.read_raw() == pytest.approx(-1000.0)
    reader.close()
    assert pin_driver.closed is True

    # Sentinel 0x7FFFFF (8388607) must be rejected
    sat_driver = FakePinDriver(0x7FFFFF)
    sat_reader = BitBangHX711Backend(sat_driver, max_read_retries=1, sleeper=lambda _: None)
    with pytest.raises(LoadCellError, match="saturasi"):
        sat_reader.read_raw()


def test_bitbang_hx711_preemption_pulse_guard():
    pin_driver = FakePinDriver(0x001000)
    times = iter([0.0, 0.0, 0.0001])  # ready ok, t_high=0.0, after read=100us (>60us)
    reader = BitBangHX711Backend(
        pin_driver,
        max_read_retries=1,
        perf_clock=lambda: next(times, 0.0002),
        sleeper=lambda _: None,
    )
    with pytest.raises(LoadCellError, match="60us"):
        reader.read_raw()


def test_loadcell_production_mode_fails_loudly_without_gpio():
    cfg = _make_loadcell_cfg(
        backend="lgpio",
        mock=LoadCellMockConfig(enabled=False, weights_g=(0.0,), static_weight_g=None, noise_g=0.0, hold_seconds_per_step=1.0),
    )
    scale = LoadCell(cfg, _pins(), environment="production")
    with pytest.raises(LoadCellError, match="Mode produksi aktif"):
        scale.init_loadcell()


def test_loadcell_auto_tare_deadband_and_mock_weight():
    clock = FakeClock()
    cfg = _make_loadcell_cfg()
    scale = LoadCell(cfg, _pins(), environment="mock", clock=clock, sleeper=lambda _: None)
    scale.init_loadcell()

    assert scale.read_weight() == 0.0

    # Within deadband (< 5g) -> returns 0.0
    scale.set_mock_weight(3.5)
    assert scale.read_weight() == 0.0

    # Real food weight -> returns accurate weight
    scale.set_mock_weight(245.5)
    assert scale.read_weight() == pytest.approx(245.5, abs=0.1)


def test_stability_monitor_restarts_on_disturbance_and_confirms_removal():
    clock = FakeClock(10.0)
    cfg = _make_loadcell_cfg(stability_duration_s=1.5, stability_tolerance_g=3.0, removal_confirm_s=0.8)
    scale = LoadCell(cfg, _pins(), environment="mock", clock=clock, sleeper=lambda _: None)
    mon = scale.stability

    # Below trigger
    s0 = mon.evaluate(5.0, clock())
    assert not s0.is_above_trigger
    assert not s0.is_stable

    # Place 300g, then disturb to 320g after 1.0s -> must reset timer
    assert not mon.evaluate(300.0, clock.advance(1.0)).is_stable
    s_jump = mon.evaluate(320.0, clock.advance(1.0))
    assert s_jump.stable_elapsed_s == 0.0
    assert not s_jump.is_stable

    # Hold steady around 320g for >= 1.5s -> stabilizes
    assert not mon.evaluate(320.5, clock.advance(0.8)).is_stable
    s_ok = mon.evaluate(319.8, clock.advance(0.8))
    assert s_ok.is_stable
    assert s_ok.weight_g == pytest.approx(320.0, abs=0.6)

    # Plate removal hysteresis (< 10g for 0.8s)
    assert not mon.is_plate_removed(15.0, clock.advance(0.2))
    assert not mon.is_plate_removed(2.0, clock.advance(0.2))
    assert mon.is_plate_removed(1.5, clock.advance(0.9))


def test_hardware_gpio_wrappers_with_fake_modules(monkeypatch):
    # Fake lgpio
    fake_lgpio = types.SimpleNamespace(
        gpiochip_open=lambda _idx: 7,
        gpio_claim_input=lambda _h, _p: None,
        gpio_claim_output=lambda _h, _p, _v: None,
        gpio_read=lambda _h, _p: 1,
        gpio_write=lambda _h, _p, _v: None,
        gpiochip_close=lambda _h: None,
    )
    monkeypatch.setitem(sys.modules, "lgpio", fake_lgpio)
    lg = LgpioPinDriver(5, 6)
    assert lg.read_dout() == 1
    lg.write_sck(1)
    lg.close()

    # Fake RPi.GPIO
    fake_rpi = types.SimpleNamespace(
        BCM=11,
        IN=1,
        OUT=0,
        LOW=0,
        HIGH=1,
        setwarnings=lambda _b: None,
        setmode=lambda _m: None,
        setup=lambda *_a, **_kw: None,
        input=lambda _p: 0,
        output=lambda _p, _v: None,
        cleanup=lambda _pins: None,
    )
    monkeypatch.setitem(sys.modules, "RPi", types.SimpleNamespace(GPIO=fake_rpi))
    monkeypatch.setitem(sys.modules, "RPi.GPIO", fake_rpi)
    rp = RpiGpioPinDriver(5, 6)
    assert rp.read_dout() == 0
    rp.write_sck(1)
    rp.close()

    # Fake adafruit_hx711 + board + digitalio
    class FakePin:
        def __init__(self, _pin):
            self.direction = None
            self.deinited = False

        def deinit(self):
            self.deinited = True

    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(D5="D5", D6="D6"))
    monkeypatch.setitem(
        sys.modules,
        "digitalio",
        types.SimpleNamespace(DigitalInOut=FakePin, Direction=types.SimpleNamespace(OUTPUT="out")),
    )
    monkeypatch.setitem(
        sys.modules,
        "adafruit_hx711",
        types.SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules,
        "adafruit_hx711.hx711",
        types.SimpleNamespace(
            HX711=type(
                "HX",
                (),
                {
                    "CHAN_A_GAIN_128": 1,
                    "CHAN_A_GAIN_64": 2,
                    "CHAN_B_GAIN_32": 3,
                    "__init__": lambda self, d, c: None,
                },
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "adafruit_hx711.analog_in",
        types.SimpleNamespace(AnalogIn=lambda _hx, _ch: types.SimpleNamespace(value=12345)),
    )
    ada = AdafruitHX711Backend(5, 6, 128)
    assert ada.read_raw() == 12345.0
    ada.close()


def test_loadcell_failsafe_reinit_on_backend_error():
    class FlakyBackend:
        def __init__(self):
            self.calls = 0

        def read_raw(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("kabel goyang")
            return 1000.0

        def close(self):
            pass

    cfg = _make_loadcell_cfg(auto_tare_on_boot=False, read_samples=1)
    scale = LoadCell(cfg, _pins(), environment="mock", backend=FlakyBackend(), sleeper=lambda _: None)
    # First read raises inside _collect_raw_samples -> triggers close() + _create_backend() fallback
    sample = scale.read_sample()
    assert sample.weight_g == 0.0
    scale.close()


def test_auto_backend_prefers_lgpio_over_adafruit(monkeypatch):
    class PoisonModule:
        def __getattr__(self, name):
            raise AssertionError(f"adafruit_hx711 must not be accessed when lgpio works ({name})")

    fake_lgpio = types.SimpleNamespace(
        gpiochip_open=lambda _idx: 7,
        gpio_claim_input=lambda _h, _p: None,
        gpio_claim_output=lambda _h, _p, _v: None,
        gpio_read=lambda _h, _p: 0,
        gpio_write=lambda _h, _p, _v: None,
        gpiochip_close=lambda _h: None,
    )
    monkeypatch.setitem(sys.modules, "lgpio", fake_lgpio)
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(D5="D5", D6="D6"))
    monkeypatch.setitem(
        sys.modules,
        "digitalio",
        types.SimpleNamespace(DigitalInOut=lambda _p: None, Direction=types.SimpleNamespace(OUTPUT="out")),
    )
    monkeypatch.setitem(sys.modules, "adafruit_hx711", PoisonModule())
    monkeypatch.setitem(sys.modules, "adafruit_hx711.hx711", PoisonModule())
    monkeypatch.setitem(sys.modules, "adafruit_hx711.analog_in", PoisonModule())

    cfg = _make_loadcell_cfg(
        backend="auto",
        auto_tare_on_boot=False,
        mock=LoadCellMockConfig(
            enabled=False,
            weights_g=(0.0,),
            static_weight_g=None,
            noise_g=0.0,
            hold_seconds_per_step=1.0,
        ),
    )
    scale = LoadCell(cfg, _pins(), environment="production")
    scale.init_loadcell()
    try:
        assert isinstance(scale._backend, BitBangHX711Backend)
    finally:
        scale.close()


def test_adafruit_backend_times_out_when_dout_stuck_high(monkeypatch):
    class FakePin:
        def __init__(self, _pin):
            self.direction = None
            self.value = True
            self.deinited = False

        def deinit(self):
            self.deinited = True

    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(D5="D5", D6="D6"))
    monkeypatch.setitem(
        sys.modules,
        "digitalio",
        types.SimpleNamespace(DigitalInOut=FakePin, Direction=types.SimpleNamespace(OUTPUT="out")),
    )
    monkeypatch.setitem(sys.modules, "adafruit_hx711", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "adafruit_hx711.hx711",
        types.SimpleNamespace(
            HX711=type(
                "HX",
                (),
                {
                    "CHAN_A_GAIN_128": 1,
                    "CHAN_A_GAIN_64": 2,
                    "CHAN_B_GAIN_32": 3,
                    "__init__": lambda self, d, c: None,
                },
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "adafruit_hx711.analog_in",
        types.SimpleNamespace(AnalogIn=lambda _hx, _ch: types.SimpleNamespace(value=12345)),
    )

    ada = AdafruitHX711Backend(5, 6, 128, read_timeout_s=0.05)
    try:
        with pytest.raises(LoadCellError, match="Timeout"):
            ada.read_raw()
    finally:
        ada.close()


def test_test_loadcell_script_mock_count():
    from scripts.test_loadcell import main as loadcell_cli_main

    rc = loadcell_cli_main(["--mock", "--count", "3", "--interval", "0.01"])
    assert rc == 0


