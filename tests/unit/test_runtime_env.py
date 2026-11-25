import io
import os
import sys

from nutriscan.core.runtime_env import (
    ARM_SAFE_ENV_DEFAULTS,
    apply_arm_runtime_guards,
    build_arm_safe_env,
    reexec_with_arm_guards_if_needed,
)


def test_build_arm_safe_env_sets_defaults_and_preserves_overrides():
    base = {
        "PATH": "/usr/bin",
        "OMP_NUM_THREADS": "4",
        "WAYLAND_DISPLAY": "wayland-0",
    }
    snapshot = dict(base)

    result = build_arm_safe_env(base, machine="aarch64", enable_gui=True)

    # Caller dictionary must never be mutated
    assert base == snapshot
    for key, val in ARM_SAFE_ENV_DEFAULTS.items():
        if key == "OMP_NUM_THREADS":
            assert result[key] == "4"
        else:
            assert result[key] == val

    assert result["QT_QPA_PLATFORM"] == "xcb"
    assert result["_NUTRISCAN_ENV_READY"] == "1"


def test_build_arm_safe_env_respects_existing_qt_platform():
    base = {
        "WAYLAND_DISPLAY": "wayland-0",
        "QT_QPA_PLATFORM": "wayland",
    }
    result = build_arm_safe_env(base, machine="aarch64", enable_gui=True)
    assert result["QT_QPA_PLATFORM"] == "wayland"

    no_gui = build_arm_safe_env({"WAYLAND_DISPLAY": "wayland-0"}, machine="aarch64", enable_gui=False)
    assert "QT_QPA_PLATFORM" not in no_gui


def test_apply_arm_runtime_guards_safe_when_stderr_has_no_fileno(monkeypatch):
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    apply_arm_runtime_guards(enable_gui=False)

    assert os.environ["OPENBLAS_CORETYPE"] == "ARMV8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"


def test_reexec_with_arm_guards_skips_non_arm_and_ready_env(monkeypatch):
    called: list[tuple] = []

    def fake_execve(path, argv, env):
        called.append((path, argv, env))

    monkeypatch.delenv("_NUTRISCAN_ENV_READY", raising=False)
    reexec_with_arm_guards_if_needed(machine="x86_64", execve_fn=fake_execve)
    assert called == []

    monkeypatch.setenv("_NUTRISCAN_ENV_READY", "1")
    reexec_with_arm_guards_if_needed(machine="aarch64", execve_fn=fake_execve)
    assert called == []


def test_reexec_with_arm_guards_execs_once_on_aarch64(monkeypatch):
    captured: dict = {}

    def fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv
        captured["env"] = env

    monkeypatch.delenv("_NUTRISCAN_ENV_READY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    reexec_with_arm_guards_if_needed(
        enable_gui=True,
        machine="aarch64",
        argv=["scripts/test_camera_ai.py", "--live"],
        execve_fn=fake_execve,
    )

    assert captured["path"] == sys.executable
    assert captured["argv"] == [sys.executable, "scripts/test_camera_ai.py", "--live"]
    assert captured["env"]["_NUTRISCAN_ENV_READY"] == "1"
    assert captured["env"]["OPENBLAS_CORETYPE"] == "ARMV8"
    assert captured["env"]["QT_QPA_PLATFORM"] == "xcb"
