"""Grafeo CLI launcher.

Thin wrapper that finds and runs the grafeo binary bundled with this package.
Install with: pip install grafeo-cli
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

__version__ = "0.5.43"

# GitHub release download URL template
_GITHUB_RELEASE = "https://github.com/GrafeoDB/grafeo/releases/download"


def _binary_name() -> str:
    """Return the platform-specific binary name."""
    return "grafeo.exe" if platform.system() == "Windows" else "grafeo"


def _find_binary() -> Path | None:
    """Find the grafeo binary.

    Search order:
    1. Bundled with this package (wheel data)
    2. Adjacent to this package (development installs)
    3. On PATH
    """
    binary = _binary_name()

    # 1. Bundled in package directory
    pkg_dir = Path(__file__).parent
    bundled = pkg_dir / binary
    if bundled.is_file():
        return bundled

    # 2. Adjacent to package (e.g., bin/ directory in virtualenv)
    for bin_dir in [pkg_dir.parent / "bin", pkg_dir.parent / "Scripts"]:
        candidate = bin_dir / binary
        if candidate.is_file():
            return candidate

    # 3. On system PATH
    from shutil import which

    on_path = which("grafeo")
    if on_path:
        return Path(on_path)

    return None


def main() -> None:
    """Run the grafeo CLI binary, forwarding all arguments."""
    binary = _find_binary()

    if binary is None:
        print(
            "error: grafeo binary not found.\n"
            "\n"
            "The grafeo-cli package is a thin launcher for the Grafeo CLI binary.\n"
            "Install the binary via one of:\n"
            f"  - Download from {_GITHUB_RELEASE}/v{__version__}/\n"
            "  - cargo install grafeo-cli\n"
            "  - Place the 'grafeo' binary on your PATH\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Make binary executable on Unix
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o111)

    try:
        result = subprocess.run(
            [str(binary), *sys.argv[1:]],
            check=False,
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError:
        print(f"error: failed to execute {binary}", file=sys.stderr)
        sys.exit(1)
