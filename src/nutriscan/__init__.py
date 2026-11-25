"""NutriScan IoT Raspberry Pi 4 package."""

from __future__ import annotations

from nutriscan.core.runtime_env import apply_arm_runtime_guards

apply_arm_runtime_guards(enable_gui=False)

__version__ = "1.0.0"
