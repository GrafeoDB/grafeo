"""Auto-discovery conftest for .gtest spec files.

Imports the pytest_collect_file hook from the Python runner so that
``pytest tests/spec/`` discovers .gtest files without needing the
``-p tests.spec.runners.python.conftest`` flag.
"""

import importlib.util
import sys
from pathlib import Path

_runner_conftest = (
    Path(__file__).resolve().parent / "runners" / "python" / "conftest.py"
)
_spec = importlib.util.spec_from_file_location(
    "_gtest_runner_conftest", _runner_conftest
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

pytest_collect_file = _mod.pytest_collect_file


def pytest_configure(config):
    """With ``GRAFEO_SPEC_ROOT`` set (the runner probes), the run collects the .gtest files
    under that directory instead of ``tests/spec``: the runner becomes a plugin, as this
    conftest's collection hook covers ``tests/spec`` only."""
    root = _mod.spec_root()
    if root is not None:
        config.pluginmanager.register(_mod, "grafeo_gtest_runner")
        config.args = [str(root)]


def pytest_terminal_summary(terminalreporter):
    """One line with the run's skipped and expected-failure counts (``gtest summary``)."""
    summary = _mod.summary_of(terminalreporter.config)
    terminalreporter.write_line(
        f"gtest summary: skipped={summary['skipped']} xfailed={summary['xfailed']}"
    )
