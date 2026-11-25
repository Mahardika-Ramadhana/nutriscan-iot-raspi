"""ARMv8-safe runtime guards preventing SIGILL and OpenMP/Qt clashes on Raspberry Pi 4."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import faulthandler
import os
import platform
import sys

ARM_ARCHITECTURES = frozenset({"aarch64", "armv7l", "armv8l"})

ARM_SAFE_ENV_DEFAULTS: dict[str, str] = {
    "PYTHONFAULTHANDLER": "1",
    "OPENBLAS_CORETYPE": "ARMV8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "2",
    "OMP_WAIT_POLICY": "PASSIVE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def build_arm_safe_env(
    existing_env: Mapping[str, str],
    machine: str | None = None,
    enable_gui: bool = False,
) -> dict[str, str]:
    """Return a new environment dictionary with ARMv8-safe defaults without mutating input."""
    _ = (machine or platform.machine()).lower()
    env = dict(existing_env)
    for key, default_val in ARM_SAFE_ENV_DEFAULTS.items():
        if not env.get(key):
            env[key] = default_val

    if enable_gui and env.get("WAYLAND_DISPLAY") and not env.get("QT_QPA_PLATFORM"):
        env["QT_QPA_PLATFORM"] = "xcb"

    env["_NUTRISCAN_ENV_READY"] = "1"
    return env


def apply_arm_runtime_guards(enable_gui: bool = False) -> None:
    """Apply safe env defaults in-process and enable Python faulthandler safely."""
    safe = build_arm_safe_env(os.environ, enable_gui=enable_gui)
    for key, val in safe.items():
        if key != "_NUTRISCAN_ENV_READY":
            os.environ.setdefault(key, val)
    try:
        if not faulthandler.is_enabled():
            faulthandler.enable()
    except Exception:
        pass


def configure_opencv_threads() -> None:
    """Limit OpenCV's internal thread pool to avoid OpenMP contention on ARM64."""
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass


def reexec_with_arm_guards_if_needed(
    enable_gui: bool = False,
    machine: str | None = None,
    argv: list[str] | None = None,
    execve_fn: Callable[[str, list[str], dict[str, str]], object] = os.execve,
) -> None:
    """Re-exec current Python process once on ARM so C libraries see ARMv8 env from start."""
    arch = (machine or platform.machine()).lower()
    if arch not in ARM_ARCHITECTURES or os.environ.get("_NUTRISCAN_ENV_READY") == "1":
        return

    safe_env = build_arm_safe_env(os.environ, machine=arch, enable_gui=enable_gui)
    cmd_args = argv if argv is not None else sys.argv
    execve_fn(sys.executable, [sys.executable, *cmd_args], safe_env)
