"""Thread-safe state machine for NutriScan automated tray scanning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any

from nutriscan.core.logger import get_logger

logger = get_logger("StateMachine")


class ScanState(str, Enum):
    """Lifecycle states for a single tray scan cycle."""

    IDLE = "IDLE"
    STABILIZING = "STABILIZING"
    PROCESSING = "PROCESSING"
    SENDING = "SENDING"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of the current system state for API & telemetry."""

    state: ScanState
    current_weight_g: float
    locked_weight_g: float | None
    stable_elapsed_s: float
    last_scan_payload: dict[str, Any] | None
    has_occlusion: bool
    last_error: str | None


class NutriScanStateMachine:
    """Thread-safe state controller adapted from DICA's StateMachine."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = ScanState.IDLE
        self._current_weight_g = 0.0
        self._locked_weight_g: float | None = None
        self._stable_elapsed_s = 0.0
        self._last_scan_payload: dict[str, Any] | None = None
        self._has_occlusion = False
        self._last_error: str | None = None

    @property
    def state(self) -> ScanState:
        with self._lock:
            return self._state

    @property
    def current_weight_g(self) -> float:
        with self._lock:
            return self._current_weight_g

    def update_weight(self, weight_g: float, stable_elapsed_s: float = 0.0) -> None:
        with self._lock:
            self._current_weight_g = float(weight_g)
            self._stable_elapsed_s = float(stable_elapsed_s)

    def begin_stabilizing(self, weight_g: float) -> bool:
        with self._lock:
            if self._state != ScanState.IDLE:
                return False
            self._state = ScanState.STABILIZING
            self._current_weight_g = float(weight_g)
            self._stable_elapsed_s = 0.0
            self._last_error = None
            logger.info("State -> STABILIZING (berat awal: %.1f g)", weight_g)
            return True

    def abort_stabilizing(self) -> bool:
        with self._lock:
            if self._state != ScanState.STABILIZING:
                return False
            self._state = ScanState.IDLE
            self._stable_elapsed_s = 0.0
            logger.info("Berat turun sebelum stabil. State -> IDLE")
            return True

    def trigger_processing(self, locked_weight_g: float | None = None) -> bool:
        with self._lock:
            if self._state not in (ScanState.IDLE, ScanState.STABILIZING):
                return False
            self._state = ScanState.PROCESSING
            self._locked_weight_g = (
                float(locked_weight_g) if locked_weight_g is not None else self._current_weight_g
            )
            self._has_occlusion = False
            self._last_error = None
            logger.info(
                "State -> PROCESSING (berat terkunci: %.1f g)",
                self._locked_weight_g,
            )
            return True

    def finish_processing(
        self,
        scan_payload: dict[str, Any],
        has_occlusion: bool = False,
    ) -> bool:
        with self._lock:
            if self._state != ScanState.PROCESSING:
                return False
            self._last_scan_payload = dict(scan_payload)
            self._has_occlusion = bool(has_occlusion)
            self._state = ScanState.SENDING
            logger.info("Inferensi selesai. State -> SENDING")
            return True

    def skip_empty_processing(self) -> bool:
        with self._lock:
            if self._state != ScanState.PROCESSING:
                return False
            self._state = ScanState.COOLDOWN
            logger.info("Tidak ada makanan terdeteksi (dilewati). State -> COOLDOWN")
            return True

    def cancel_processing(self, error_msg: str = "") -> bool:
        with self._lock:
            if self._state != ScanState.PROCESSING:
                return False
            self._last_error = error_msg or None
            # Enter COOLDOWN when an error occurred so a stationary plate doesn't re-trigger immediately
            self._state = ScanState.COOLDOWN if error_msg else ScanState.IDLE
            logger.warning("Pemrosesan dibatalkan (%s). State -> %s", error_msg or "manual", self._state.value)
            return True

    def finish_sending(self) -> bool:
        with self._lock:
            if self._state != ScanState.SENDING:
                return False
            self._state = ScanState.COOLDOWN
            logger.info("Payload tersimpan/terkirim. State -> COOLDOWN (tunggu piring diangkat)")
            return True

    def complete_cooldown(self) -> bool:
        with self._lock:
            if self._state != ScanState.COOLDOWN:
                return False
            self._locked_weight_g = None
            self._stable_elapsed_s = 0.0
            self._state = ScanState.IDLE
            logger.info("Piring telah diangkat. State -> IDLE")
            return True

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            payload_copy = dict(self._last_scan_payload) if self._last_scan_payload is not None else None
            return StateSnapshot(
                state=self._state,
                current_weight_g=self._current_weight_g,
                locked_weight_g=self._locked_weight_g,
                stable_elapsed_s=self._stable_elapsed_s,
                last_scan_payload=payload_copy,
                has_occlusion=self._has_occlusion,
                last_error=self._last_error,
            )
