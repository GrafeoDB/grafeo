"""Exercise the real benchmark comparator without contacting GitHub."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench-compare.sh"
BASH = shutil.which("bash")


class BenchmarkReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="benchmark-report-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.config = self.root / "thresholds.toml"
        self.summary = self.root / "job summary.md"
        self.summary.write_text("Earlier job output.\n\n")
        self.capture = self.root / "comment.json"
        self.comparison = self.root / "comparison.txt"
        self.memory = self.root / "target" / "criterion"
        self.memory.mkdir(parents=True)
        self.temporary = self.root / "temporary"
        self.temporary.mkdir()
        self.write_tool(
            "critcmp",
            """import os, pathlib, sys
if sys.argv[1:] != ["base", "pr", "--color", "never"]:
    raise SystemExit("unexpected comparison arguments")
print(pathlib.Path(os.environ["COMPARISON_FIXTURE"]).read_text(), end="")
""",
        )
        self.write_tool(
            "gh",
            """import json, os, pathlib, sys
args = sys.argv[1:]
if args[:2] == ["api", "repos/GrafeoDB/grafeo/issues/609/comments"]:
    print(os.environ["EXISTING_COMMENT"])
    raise SystemExit(0)
if args[:4] == ["api", "--method", "PATCH", "repos/GrafeoDB/grafeo/issues/comments/123"]:
    if len(args) != 6 or args[4] != "-f" or not args[5].startswith("body="):
        raise SystemExit("unexpected update arguments")
    body, method = args[5][5:], "update"
elif args[:4] == ["pr", "comment", "609", "--body"] and len(args) == 5:
    body, method = args[4], "new"
else:
    raise SystemExit("unexpected GitHub arguments")
pathlib.Path(os.environ["COMMENT_CAPTURE"]).write_text(json.dumps({"body": body, "method": method}))
if os.environ["ALLOW_COMMENT"] != "yes":
    print("GraphQL: Resource not accessible by integration (addComment)", file=sys.stderr)
    raise SystemExit(1)
""",
        )

    def write_tool(self, name, source):
        path = self.bin / name
        path.write_text("#!/usr/bin/env python3\n" + source)
        path.chmod(0o755)

    def run_comparison(
        self,
        *,
        existing=False,
        allowed=False,
        core=False,
        memory=False,
        query=False,
        memory_blocks=True,
        core_threshold=30,
        summary=True,
    ):
        self.config.write_text(f"""[defaults]
threshold_pct = 15
fail_ci = false
[categories.core]
threshold_pct = {core_threshold}
fail_ci = true
benchmarks = ["core_*"]
[categories.query]
threshold_pct = 12
fail_ci = false
benchmarks = ["query_*"]
[memory]
fail_ci = {str(memory_blocks).lower()}
[memory.bounds]
memory_fixture = 100
""")
        core_ratio = "1.31" if core else "1.00"
        query_ratio = "1.13" if query else "1.00"
        self.comparison.write_text(
            "group base pr\n-----\n"
            f"core_fixture 1.00 100 ns {core_ratio} 131 ns\n"
            f"query_fixture 1.00 100 ns {query_ratio} 113 ns\n"
        )
        (self.memory / "memory_snapshot_base.json").write_text(
            json.dumps({"memory_fixture": 40})
        )
        (self.memory / "memory_snapshot.json").write_text(
            json.dumps({"memory_fixture": 200 if memory else 50})
        )
        self.capture.unlink(missing_ok=True)
        env = os.environ.copy()
        env.pop("GITHUB_STEP_SUMMARY", None)
        env.update(
            {
                "PATH": str(self.bin) + os.pathsep + env["PATH"],
                "GITHUB_REPOSITORY": "GrafeoDB/grafeo",
                "COMPARISON_FIXTURE": str(self.comparison),
                "COMMENT_CAPTURE": str(self.capture),
                "EXISTING_COMMENT": "123" if existing else "",
                "ALLOW_COMMENT": "yes" if allowed else "no",
                "TMPDIR": str(self.temporary),
            }
        )
        if summary is True:
            env["GITHUB_STEP_SUMMARY"] = str(self.summary)
        elif summary == "empty":
            env["GITHUB_STEP_SUMMARY"] = ""
        return subprocess.run(
            [BASH, str(SCRIPT), "base", "pr", str(self.config), "609"],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def assert_report(self, result, *, existing, allowed, core, memory):
        attempted = json.loads(self.capture.read_text())
        body = attempted["body"]
        self.assertEqual(attempted["method"], "update" if existing else "new")
        self.assertEqual(
            self.summary.read_text(), "Earlier job output.\n\n\n" + body + "\n"
        )
        self.assertIn("## Benchmark Comparison", body)
        self.assertIn(self.comparison.read_text().rstrip(), body)
        self.assertIn("### Memory usage", body)
        self.assertIn("EXCEEDED" if memory else "OK", body)
        self.assertTrue(body.endswith("<!-- grafeo-bench-comparison -->"))
        if core:
            self.assertIn("+31.0%", body)
            self.assertIn("**BLOCKING**", body)
        else:
            self.assertIn(
                "No performance regressions above configured thresholds.", body
            )
        if allowed:
            self.assertIn(
                "Updated existing comment" if existing else "Posted new comment",
                result.stdout,
            )
        else:
            self.assertIn("warning", result.stderr.lower())
            self.assertNotIn("Posted new comment", result.stdout)
            self.assertNotIn("Updated existing comment", result.stdout)

    def test_denied_comment_keeps_passing_verdict_and_complete_summary(self):
        result = self.run_comparison()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("All benchmarks within configured thresholds.", result.stdout)
        self.assert_report(
            result, existing=False, allowed=False, core=False, memory=False
        )

    def test_comment_permissions_do_not_change_performance_or_memory_verdict(self):
        for existing in [False, True]:
            for allowed in [False, True]:
                for core, memory in [
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ]:
                    with self.subTest(
                        existing=existing, allowed=allowed, core=core, memory=memory
                    ):
                        self.summary.write_text("Earlier job output.\n\n")
                        result = self.run_comparison(
                            existing=existing, allowed=allowed, core=core, memory=memory
                        )
                        self.assertEqual(
                            result.returncode, int(core or memory), result.stderr
                        )
                        if core or memory:
                            self.assertIn(
                                f"ERROR: {int(core)} blocking regression(s), {int(memory)} memory bound failure(s)",
                                result.stdout,
                            )
                            self.assertNotIn(
                                "All benchmarks within configured thresholds.",
                                result.stdout,
                            )
                        else:
                            self.assertIn(
                                "All benchmarks within configured thresholds.",
                                result.stdout,
                            )
                        self.assert_report(
                            result,
                            existing=existing,
                            allowed=allowed,
                            core=core,
                            memory=memory,
                        )

    def test_nonblocking_query_and_memory_warnings_remain_nonblocking(self):
        result = self.run_comparison(query=True, memory=True, memory_blocks=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(self.capture.read_text())["body"]
        self.assertIn("+13.0%", body)
        self.assertIn("EXCEEDED", body)
        self.assertNotIn("**BLOCKING**", body)
        self.assertIn("All benchmarks within configured thresholds.", result.stdout)
        self.assertEqual(
            self.summary.read_text(), "Earlier job output.\n\n\n" + body + "\n"
        )

    def test_configured_threshold_override_is_preserved(self):
        result = self.run_comparison(core=True, core_threshold=40)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("All benchmarks within configured thresholds.", result.stdout)
        self.assertIn(
            "No performance regressions above configured thresholds.",
            self.summary.read_text(),
        )

    def test_existing_summary_without_newline_keeps_report_heading(self):
        self.summary.write_text("Earlier job output.")
        result = self.run_comparison(allowed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "## Benchmark Comparison (base vs PR)",
            self.summary.read_text().splitlines(),
        )
        self.assertTrue(self.summary.read_text().startswith("Earlier job output.\n"))

    def test_unset_summary_still_reaches_final_verdict(self):
        for summary in [False, "empty"]:
            with self.subTest(summary=summary):
                result = self.run_comparison(summary=summary)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    "All benchmarks within configured thresholds.", result.stdout
                )
                self.assertEqual(self.summary.read_text(), "Earlier job output.\n\n")

    def test_summary_write_failure_is_fatal(self):
        self.summary.unlink()
        self.summary.mkdir()
        result = self.run_comparison(allowed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("All benchmarks within configured thresholds.", result.stdout)
        self.assertEqual(list(self.temporary.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
