"""HX711 Load Cell driver, outlier filtering, stability monitor, and mock simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Callable, Protocol

from nutriscan.core.config import HardwarePinsConfig, LoadCellConfig
from nutriscan.core.logger import get_logger

logger = get_logger("LoadCell")

# Nilai sentinel saturasi/error 24-bit signed HX711
_HX711_INVALID_SENTINELS = frozenset({-8388608, 8388607, 0, -1})
# Batas pulsa PD_SCK HIGH (60 mikrodetik) sebelum HX711 masuk power-down
_HX711_MAX_PULSE_S = 60e-6
_GAIN_EXTRA_PULSES = {128: 1, 64: 3, 32: 2}


class LoadCellError(RuntimeError):
    """Raised when HX711 initialization or reading fails."""


@dataclass(frozen=True)
class WeightSample:
    """Single filtered weight reading paired with raw ADC average."""

    weight_g: float
    raw_value: float
    timestamp: float


@dataclass(frozen=True)
class StabilityStatus:
    """Evaluated stability state for automated plate detection."""

    weight_g: float
    is_above_trigger: bool
    is_stable: bool
    stable_elapsed_s: float
    spread_g: float


class HX711Backend(Protocol):
    """Abstract low-level raw ADC backend for HX711."""

    def read_raw(self) -> float:
        """Return a single 24-bit raw ADC value."""

    def close(self) -> None:
        """Release underlying GPIO resources."""


class GpioPinDriver(Protocol):
    """Minimal GPIO pin read/write interface for bit-bang HX711."""

    def read_dout(self) -> int: ...

    def write_sck(self, level: int) -> None: ...

    def close(self) -> None: ...


class LgpioPinDriver:
    """Raspberry Pi OS Bookworm+ GPIO backend via lgpio."""

    def __init__(self, dout_pin: int, sck_pin: int, chip_index: int = 0) -> None:
        import lgpio

        self._lgpio = lgpio
        self._dout_pin = dout_pin
        self._sck_pin = sck_pin
        self._handle = lgpio.gpiochip_open(chip_index)
        lgpio.gpio_claim_input(self._handle, dout_pin)
        lgpio.gpio_claim_output(self._handle, sck_pin, 0)

    def read_dout(self) -> int:
        return int(self._lgpio.gpio_read(self._handle, self._dout_pin))

    def write_sck(self, level: int) -> None:
        self._lgpio.gpio_write(self._handle, self._sck_pin, 1 if level else 0)

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._lgpio.gpiochip_close(self._handle)
            finally:
                self._handle = None


class RpiGpioPinDriver:
    """Fallback GPIO backend via RPi.GPIO (legacy Raspberry Pi OS)."""

    def __init__(self, dout_pin: int, sck_pin: int) -> None:
        import RPi.GPIO as GPIO

        self._gpio = GPIO
        self._dout_pin = dout_pin
        self._sck_pin = sck_pin
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(dout_pin, GPIO.IN)
        GPIO.setup(sck_pin, GPIO.OUT, initial=GPIO.LOW)

    def read_dout(self) -> int:
        return int(self._gpio.input(self._dout_pin))

    def write_sck(self, level: int) -> None:
        self._gpio.output(self._sck_pin, self._gpio.HIGH if level else self._gpio.LOW)

    def close(self) -> None:
        self._gpio.cleanup([self._dout_pin, self._sck_pin])


class AdafruitHX711Backend:
    """HX711 backend using adafruit_circuitpython_hx711 with bounded DOUT timeout."""

    def __init__(
        self,
        dout_pin: int,
        sck_pin: int,
        gain: int = 128,
        read_timeout_s: float = 0.4,
        perf_clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        import board
        import digitalio
        from adafruit_hx711.analog_in import AnalogIn
        from adafruit_hx711.hx711 import HX711

        dt_obj = getattr(board, f"D{dout_pin}", None)
        sck_obj = getattr(board, f"D{sck_pin}", None)
        if dt_obj is None or sck_obj is None:
            raise LoadCellError(f"Pin board D{dout_pin} atau D{sck_pin} tidak tersedia")

        self._data = digitalio.DigitalInOut(dt_obj)
        self._clock = digitalio.DigitalInOut(sck_obj)
        self._clock.direction = digitalio.Direction.OUTPUT
        if hasattr(self._clock, "value"):
            self._clock.value = False
        self._hx = HX711(self._data, self._clock)

        gain_map = {
            128: HX711.CHAN_A_GAIN_128,
            64: HX711.CHAN_A_GAIN_64,
            32: HX711.CHAN_B_GAIN_32,
        }
        channel_const = gain_map.get(gain, HX711.CHAN_A_GAIN_128)
        self._analog_in = AnalogIn(self._hx, channel_const)
        self._read_timeout_s = read_timeout_s
        self._perf_clock = perf_clock
        self._sleeper = sleeper

    def read_raw(self) -> float:
        if hasattr(self._clock, "value"):
            self._clock.value = False
        if hasattr(self._data, "value"):
            deadline = self._perf_clock() + self._read_timeout_s
            while self._data.value:
                if self._perf_clock() >= deadline:
                    raise LoadCellError("Timeout menunggu HX711 DOUT menjadi LOW (cek kabel VCC/GND/DT/SCK)")
                self._sleeper(0.001)
        return float(self._analog_in.value)

    def close(self) -> None:
        for pin in (self._data, self._clock):
            try:
                pin.deinit()
            except Exception:
                pass


class BitBangHX711Backend:
    """Pure-Python 24-bit HX711 serial reader with Linux preemption guard."""

    def __init__(
        self,
        pin_driver: GpioPinDriver,
        gain: int = 128,
        read_timeout_s: float = 0.4,
        max_read_retries: int = 3,
        perf_clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if gain not in _GAIN_EXTRA_PULSES:
            raise LoadCellError(f"Gain HX711 tidak didukung: {gain}")
        self._pins = pin_driver
        self._extra_pulses = _GAIN_EXTRA_PULSES[gain]
        self._read_timeout_s = read_timeout_s
        self._max_read_retries = max_read_retries
        self._perf_clock = perf_clock
        self._sleeper = sleeper

    def _wait_ready(self) -> None:
        self._pins.write_sck(0)
        deadline = self._perf_clock() + self._read_timeout_s
        while self._pins.read_dout() != 0:
            if self._perf_clock() >= deadline:
                raise LoadCellError("Timeout menunggu HX711 DOUT menjadi LOW (cek kabel DT/SCK)")
            self._sleeper(0.001)

    def _read_single_attempt(self) -> int:
        self._wait_ready()
        raw = 0
        for _ in range(24):
            t_high = self._perf_clock()
            self._pins.write_sck(1)
            bit = self._pins.read_dout()
            self._pins.write_sck(0)
            if (self._perf_clock() - t_high) > _HX711_MAX_PULSE_S:
                raise LoadCellError("Pulsa SCK melebihi 60us akibat preemptive scheduling")
            raw = (raw << 1) | (1 if bit else 0)

        for _ in range(self._extra_pulses):
            self._pins.write_sck(1)
            self._pins.write_sck(0)

        if raw & 0x800000:
            raw -= 0x1000000
        if raw in _HX711_INVALID_SENTINELS:
            raise LoadCellError(f"Nilai saturasi/putus HX711 terdeteksi: {raw}")
        return raw

    def read_raw(self) -> float:
        last_err: Exception | None = None
        for _ in range(self._max_read_retries):
            try:
                return float(self._read_single_attempt())
            except LoadCellError as exc:
                last_err = exc
                self._sleeper(0.01)
        raise LoadCellError(f"Gagal membaca HX711 setelah {self._max_read_retries} percobaan: {last_err}")

    def close(self) -> None:
        self._pins.close()


class MockHX711Backend:
    """Deterministic HX711 simulator producing raw ADC counts from configured gram steps."""

    def __init__(
        self,
        config: LoadCellConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._clock = clock
        self._start_ts = clock()
        self._manual_weight_g: float | None = None
        self._sample_idx = 0

    def set_weight(self, weight_g: float | None) -> None:
        """Force a specific simulated weight in grams (or None to resume timeline)."""
        self._manual_weight_g = float(weight_g) if weight_g is not None else None

    def _target_weight_g(self) -> float:
        if self._manual_weight_g is not None:
            return self._manual_weight_g
        if self._config.mock.static_weight_g is not None:
            return self._config.mock.static_weight_g
        elapsed = max(0.0, self._clock() - self._start_ts)
        steps = self._config.mock.weights_g
        idx = int(elapsed / self._config.mock.hold_seconds_per_step)
        return steps[min(idx, len(steps) - 1)]

    def read_raw(self) -> float:
        base_g = self._target_weight_g()
        noise = 0.0
        if self._config.mock.noise_g > 0:
            # Deterministic small oscillation avoids flaky tests while mimicking sensor jitter
            noise = self._config.mock.noise_g * math.sin(self._sample_idx * 1.7)
        self._sample_idx += 1
        simulated_g = max(0.0, base_g + noise) if base_g <= 0.0 else (base_g + noise)
        return self._config.tare_offset + simulated_g * self._config.reference_unit

    def close(self) -> None:
        return None


def trimmed_mean(samples: list[float]) -> float:
    """Compute trimmed mean dropping 2 lowest & 2 highest outliers when len > 4 (DICA pattern)."""
    if not samples:
        raise LoadCellError("Tidak ada sampel untuk dihitung rata-ratanya")
    ordered = sorted(samples)
    valid = ordered[2:-2] if len(ordered) > 4 else ordered
    return float(sum(valid) / len(valid))


class WeightStabilityMonitor:
    """Tracks weight samples over time to confirm stable load and plate removal."""

    def __init__(self, config: LoadCellConfig) -> None:
        self._config = config
        self._stable_since: float | None = None
        self._window: deque[float] = deque(maxlen=max(3, config.read_samples))
        self._removal_since: float | None = None

    def reset(self) -> None:
        self._stable_since = None
        self._window.clear()
        self._removal_since = None

    def evaluate(self, weight_g: float, now: float) -> StabilityStatus:
        """Evaluate whether a plate load above trigger threshold has stabilized."""
        if weight_g < self._config.trigger_threshold_g:
            self._stable_since = None
            self._window.clear()
            return StabilityStatus(
                weight_g=weight_g,
                is_above_trigger=False,
                is_stable=False,
                stable_elapsed_s=0.0,
                spread_g=0.0,
            )

        self._window.append(weight_g)
        med = float(median(self._window))
        spread = float(max(self._window) - min(self._window))
        max_dev = max(abs(val - med) for val in self._window)

        if max_dev > self._config.stability_tolerance_g:
            # Weight shifted beyond tolerance; restart stability window from current reading
            self._window.clear()
            self._window.append(weight_g)
            self._stable_since = now
            return StabilityStatus(
                weight_g=weight_g,
                is_above_trigger=True,
                is_stable=False,
                stable_elapsed_s=0.0,
                spread_g=0.0,
            )

        if self._stable_since is None:
            self._stable_since = now

        elapsed = max(0.0, now - self._stable_since)
        is_stable = elapsed >= self._config.stability_duration_s
        return StabilityStatus(
            weight_g=round(med, 1) if is_stable else weight_g,
            is_above_trigger=True,
            is_stable=is_stable,
            stable_elapsed_s=elapsed,
            spread_g=round(spread, 2),
        )

    def is_plate_removed(self, weight_g: float, now: float) -> bool:
        """Confirm weight remained below removal_threshold_g for removal_confirm_s."""
        if weight_g >= self._config.removal_threshold_g:
            self._removal_since = None
            return False
        if self._removal_since is None:
            self._removal_since = now
        return (now - self._removal_since) >= self._config.removal_confirm_s


class LoadCell:
    """High-level HX711 Load Cell manager with auto-tare, filtering, and fail-safe recovery."""

    def __init__(
        self,
        config: LoadCellConfig,
        pins: HardwarePinsConfig,
        environment: str = "development",
        backend: HX711Backend | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._pins = pins
        self._environment = environment.lower()
        self._backend: HX711Backend | None = backend
        self._clock = clock
        self._sleeper = sleeper
        self.offset = config.tare_offset
        self.scale = config.reference_unit
        self._is_mock = backend is not None and isinstance(backend, MockHX711Backend)
        self.stability = WeightStabilityMonitor(config)

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def __enter__(self) -> LoadCell:
        self.init_loadcell()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_backend(self) -> tuple[HX711Backend, bool]:
        if self._config.backend == "mock" or self._config.mock.enabled:
            return MockHX711Backend(self._config, clock=self._clock), True

        order = (
            [self._config.backend]
            if self._config.backend != "auto"
            else ["lgpio", "adafruit_hx711", "rpi_gpio"]
        )
        errors: list[str] = []
        for choice in order:
            try:
                if choice == "adafruit_hx711":
                    return (
                        AdafruitHX711Backend(
                            self._pins.pin_loadcell_dt,
                            self._pins.pin_loadcell_sck,
                            self._config.gain,
                        ),
                        False,
                    )
                if choice == "lgpio":
                    pin_drv = LgpioPinDriver(self._pins.pin_loadcell_dt, self._pins.pin_loadcell_sck)
                    return BitBangHX711Backend(pin_drv, gain=self._config.gain), False
                if choice == "rpi_gpio":
                    pin_drv = RpiGpioPinDriver(self._pins.pin_loadcell_dt, self._pins.pin_loadcell_sck)
                    return BitBangHX711Backend(pin_drv, gain=self._config.gain), False
            except Exception as exc:
                errors.append(f"{choice}: {exc}")

        if self._environment == "production":
            raise LoadCellError(
                "Mode produksi aktif tetapi driver HX711/GPIO gagal diinisialisasi: "
                + "; ".join(errors)
            )
        logger.warning(
            "Driver HX711 tidak tersedia (%s). Beralih ke MockHX711Backend (mode %s).",
            "; ".join(errors),
            self._environment,
        )
        return MockHX711Backend(self._config, clock=self._clock), True

    def init_loadcell(self) -> None:
        """Initialize HX711 backend and optionally run auto-tare."""
        if self._backend is None:
            self._backend, self._is_mock = self._create_backend()
        logger.info(
            "LoadCell siap (backend=%s, DT=GPIO%d, SCK=GPIO%d, ref_unit=%.3f)",
            "mock" if self._is_mock else self._config.backend,
            self._pins.pin_loadcell_dt,
            self._pins.pin_loadcell_sck,
            self.scale,
        )
        if self._config.auto_tare_on_boot:
            self.tare()

    def _collect_raw_samples(self, count: int) -> list[float]:
        if self._backend is None:
            raise LoadCellError("LoadCell belum diinisialisasi; panggil init_loadcell() dulu")
        samples: list[float] = []
        last_exc: Exception | None = None
        for idx in range(count):
            try:
                samples.append(self._backend.read_raw())
            except Exception as exc:
                last_exc = exc
                logger.warning("Gagal baca sampel HX711 ke-%d: %s", idx + 1, exc)
                if not samples and idx >= 1:
                    break
            if idx + 1 < count and not self._is_mock:
                self._sleeper(0.02)
        if not samples:
            raise LoadCellError(f"Semua pembacaan sampel HX711 gagal: {last_exc}")
        return samples

    def read_raw_average(self, samples: int | None = None) -> float:
        """Return trimmed mean of raw HX711 ADC counts (for calibration & diagnostics)."""
        count = samples if samples is not None else self._config.read_samples
        raw_list = self._collect_raw_samples(max(1, count))
        return trimmed_mean(raw_list)

    def tare(self, samples: int | None = None) -> float:
        """Zero the scale using current tray load via trimmed-mean filtering."""
        count = samples if samples is not None else self._config.tare_samples
        logger.info("Melakukan auto-tare (%d sampel)...", count)
        self.offset = self.read_raw_average(count)
        self.stability.reset()
        logger.info("Tare selesai. Offset nol = %.2f", self.offset)
        return self.offset

    def read_sample(self) -> WeightSample:
        """Read current weight in grams with trimmed-mean filter, deadband, and fail-safe re-init."""
        try:
            avg_raw = self.read_raw_average(self._config.read_samples)
        except LoadCellError as exc:
            logger.error("Kegagalan pembacaan HX711 (%s); mencoba re-inisialisasi...", exc)
            self.close()
            self._backend, self._is_mock = self._create_backend()
            avg_raw = self.read_raw_average(self._config.read_samples)

        weight = (avg_raw - self.offset) / self.scale
        if weight < self._config.deadband_g:
            weight = 0.0
        return WeightSample(
            weight_g=round(weight, 2),
            raw_value=avg_raw,
            timestamp=self._clock(),
        )

    def read_weight(self) -> float:
        """Convenience method matching DICA LoadCell.read_weight() -> grams."""
        return self.read_sample().weight_g

    def set_mock_weight(self, weight_g: float | None) -> None:
        """Override simulated weight when using MockHX711Backend."""
        if isinstance(self._backend, MockHX711Backend):
            self._backend.set_weight(weight_g)
        else:
            raise LoadCellError("set_mock_weight hanya tersedia pada mode mock")

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None
