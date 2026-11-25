"""Root entrypoint for NutriScan MBG Raspberry Pi 4 runtime."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nutriscan.core.runtime_env import (  # noqa: E402
    apply_arm_runtime_guards,
    reexec_with_arm_guards_if_needed,
)

if __name__ == "__main__":
    reexec_with_arm_guards_if_needed(enable_gui=False)

apply_arm_runtime_guards(enable_gui=False)

from nutriscan.core.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
