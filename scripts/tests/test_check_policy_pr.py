"""Tests for `scripts/check_policy.py pr` (pull request eligibility) and the commit-msg patterns.

Run: uv run --with pytest python -m pytest scripts/tests
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_check_policy import found, git, policy, write

EM = chr(0x2014)
BOX = "- [x] I have read every line of this change, I understand it, and I can explain it."
PLANNED = {"milestone": {"title": "0.5.44"}, "labels": [], "type": {"name": "Feature"}}
BUG = {"milestone": {"title": "0.5.44"}, "labels": [], "type": {"name": "Bug"}}
UNPLANNED = {"milestone": None, "labels": [], "type": None}
FIRST_ISSUE = {
    "milestone": None,
    "labels": [{"name": "good first issue"}],
    "type": None,
}
TRAILER = ("feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>",)


def test_commit_message_made_with_cursor(tmp_path: Path) -> None:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(
        "feat: x\n\nMade with [Cursor](https://cursor.com)\n", encoding="utf-8"
    )
    assert found(policy(tmp_path, "commit-msg", str(path))) == ["P3 COMMIT_EDITMSG:3"]
    path.write_text("fix: rows made with an open cursor were lost\n", encoding="utf-8")
    assert found(policy(tmp_path, "commit-msg", str(path))) == []


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """`main` and `release/0.5.44` at the same base commit, and a `topic` branch to change."""
    root = tmp_path / "pr"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write(
        root,
        "crates/grafeo-core/Cargo.toml",
        '[package]\nname = "grafeo-core"\n\n[dependencies]\nthiserror = "2"\n\n'
        "[features]\ndefault = []\n",
    )
    write(root, "crates/grafeo-core/src/lib.rs", "pub fn a() {}\n")
    write(
        root, "crates/bindings/node/package.json", '{"dependencies": {"napi": "2"}}\n'
    )
    write(root, ".github/workflows/ci.yml", "name: CI\n")
    write(root, "CHANGELOG.md", "# Changelog\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "base")
    git(root, "branch", "release/0.5.44")
    git(root, "switch", "-q", "-c", "topic", "release/0.5.44")
    return root


def commit(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        write(repo, rel, text)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "change")


def check(
    repo: Path,
    *,
    author: str = "CONTRIBUTOR",
    login: str = "vincent",
    base: str = "release/0.5.44",
    head_ref: str = "topic",
    title: str = "feat: add a thing",
    body: str = "Fixes #3",
    labels: tuple[str, ...] = (),
    commits: tuple[str, ...] = ("feat: add a thing",),
    issues: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `pr` against the `topic` branch with a hand-built GitHub context."""
    same_repo = author != "CONTRIBUTOR"
    bot = login.endswith("[bot]")
    context = {
        "pull": {
            "number": 7,
            "title": title,
            "body": body,
            "user": {"login": login, "type": "Bot" if bot else "User"},
            "author_association": author,
            "labels": [{"name": label} for label in labels],
            "base": {"ref": base, "repo": {"full_name": "GrafeoDB/grafeo"}},
            "head": {
                "ref": head_ref,
                "repo": {
                    "full_name": "GrafeoDB/grafeo" if same_repo else f"{login}/grafeo"
                },
            },
        },
        "commits": list(commits),
        "issues": {"3": PLANNED} if issues is None else issues,
    }
    path = repo.parent / "context.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    args = [
        "pr",
        "--number",
        "7",
        "--head",
        "topic",
        "--base",
        base,
        "--context",
        str(path),
    ]
    return policy(repo, *args)


def test_eligible_contributor_pr_passes(repo: Path) -> None:
    commit(
        repo,
        {
            "crates/grafeo-core/src/lib.rs": "pub fn a() {}\npub fn b() {}\n",
            "crates/grafeo-core/tests/b.rs": "#[test]\nfn b() {}\n",
            "CHANGELOG.md": "# Changelog\n- b\n",
        },
    )
    result = check(repo)
    assert (result.returncode, found(result)) == (0, [])
    assert "PR #7 by vincent (external)" in result.stderr


def test_p1_target_branch(repo: Path) -> None:
    commit(repo, {"README.md": "x\n"})
    assert found(check(repo, base="main")) == ["P1 PR"]
    maintainer = check(repo, base="main", author="MEMBER", body="")
    assert (maintainer.returncode, found(maintainer)) == (0, ["P1 PR"])
    release = check(
        repo, base="main", author="OWNER", head_ref="release/0.5.44", body=""
    )
    assert found(release) == []


@pytest.mark.parametrize(
    ("body", "issues", "expected"),
    [
        ("", {}, ["P2 PR"]),
        ("Fixes #4", {"4": UNPLANNED}, ["P2 PR"]),
        ("Closes https://github.com/GrafeoDB/grafeo/issues/4", {"4": FIRST_ISSUE}, []),
        ("Refs #3", {"3": PLANNED}, []),
        ("<!-- Fixes #3 -->", {}, ["P2 PR"]),
    ],
)
def test_p2_planned_issue(
    repo: Path, body: str, issues: dict, expected: list[str]
) -> None:
    commit(repo, {"README.md": "x\n"})
    assert found(check(repo, body=body, issues=issues)) == expected


def test_p2_does_not_apply_to_maintainers(repo: Path) -> None:
    commit(repo, {"README.md": "x\n"})
    assert found(check(repo, author="COLLABORATOR", body="", issues={})) == []


def test_p3_ai_lines_need_the_ownership_box(repo: Path) -> None:
    commit(repo, {"README.md": "x\n"})
    unticked = check(repo, commits=TRAILER)
    assert (unticked.returncode, found(unticked)) == (1, ["P3 PR"])
    ticked = check(repo, commits=TRAILER, body=f"Fixes #3\n\n{BOX}")
    assert (ticked.returncode, found(ticked)) == (0, ["P3 PR"])
    assert "P3 PR warning" in ticked.stdout
    cursor = check(repo, body="Fixes #3\n\nMade with [Cursor](https://cursor.com)")
    assert (cursor.returncode, found(cursor)) == (1, ["P3 PR"])
    hidden = check(repo, body="Fixes #3\n<!-- Generated with [Claude Code](x) -->")
    assert found(hidden) == []
    maintainer = check(repo, author="MEMBER", commits=TRAILER)
    assert (maintainer.returncode, found(maintainer)) == (0, ["P3 PR"])


def test_p4_new_dependencies(repo: Path) -> None:
    manifest = (repo / "crates/grafeo-core/Cargo.toml").read_text()
    commit(
        repo,
        {
            "crates/grafeo-core/Cargo.toml": manifest.replace(
                'thiserror = "2"', 'thiserror = "2.1"\nserde = "1"'
            ),
            "crates/bindings/node/package.json": '{"dependencies": {"napi": "3", "pad": "1"}}\n',
            "CHANGELOG.md": "# Changelog\n- deps\n",
        },
    )
    result = check(repo)
    assert found(result) == [
        "P4 crates/bindings/node/package.json",
        "P4 crates/grafeo-core/Cargo.toml",
    ]
    assert "adds dependencies.serde;" in result.stdout
    assert "napi" not in result.stdout, "a version bump is not a new dependency"
    approved = check(repo, labels=("approved: deps",))
    assert approved.returncode == 0
    assert "(approved)" in approved.stdout
    assert check(repo, author="MEMBER").returncode == 0
    assert found(check(repo, login="dependabot[bot]", author="NONE")) == []


def test_p5_protected_paths(repo: Path) -> None:
    commit(repo, {".github/workflows/ci.yml": "name: CI\non: push\n"})
    assert found(check(repo)) == ["P5 PR"]
    approved = check(repo, labels=("approved: infra",))
    assert (approved.returncode, found(approved)) == (0, ["P5 PR"])
    assert found(check(repo, author="MEMBER")) == []


def test_p6_new_crate_and_feature(repo: Path) -> None:
    manifest = (repo / "crates/grafeo-core/Cargo.toml").read_text()
    commit(
        repo,
        {
            "crates/grafeo-new/Cargo.toml": '[package]\nname = "grafeo-new"\n',
            "crates/grafeo-core/Cargo.toml": manifest + "turbo = []\n",
            "CHANGELOG.md": "# Changelog\n- turbo\n",
        },
    )
    assert found(check(repo)) == [
        "P6 crates/grafeo-core/Cargo.toml",
        "P6 crates/grafeo-new/Cargo.toml",
    ]
    maintainer = check(repo, author="MEMBER")
    assert (maintainer.returncode, len(found(maintainer))) == (0, 2)


def test_d5_store_wrapper_needs_arch_approval_from_everyone(repo: Path) -> None:
    commit(
        repo,
        {
            "crates/grafeo-core/src/wrap.rs": "impl GraphStore for Wrapper {}\n",
            "CHANGELOG.md": "# Changelog\n- wrapper\n",
        },
    )
    maintainer = check(repo, author="MEMBER")
    assert (maintainer.returncode, found(maintainer)) == (
        1,
        ["D5 crates/grafeo-core/src/wrap.rs:1"],
    )
    assert check(repo, author="MEMBER", labels=("approved: arch",)).returncode == 0


def test_p7_fixes_need_a_test(repo: Path) -> None:
    commit(
        repo,
        {
            "crates/grafeo-core/src/lib.rs": "pub fn a() { fixed(); }\n",
            "CHANGELOG.md": "# Changelog\n- fix\n",
        },
    )
    assert found(check(repo, title="fix: a")) == ["P7 PR"]
    assert found(check(repo, title="feat: a", body="Fixes #5", issues={"5": BUG})) == [
        "P7 PR"
    ]
    assert found(check(repo, title="fix: a", author="MEMBER")) == ["P7 PR"]
    assert found(check(repo, title="fix: a", labels=("no-test-needed",))) == []
    assert found(check(repo, title="feat: a")) == []


def test_p7_an_added_test_attribute_counts(repo: Path) -> None:
    commit(
        repo,
        {
            "crates/grafeo-core/src/lib.rs": (
                "pub fn a() {}\n#[cfg(test)]\nmod t {\n    #[test]\n    fn a() {}\n}\n"
            ),
            "CHANGELOG.md": "# Changelog\n- fix\n",
        },
    )
    assert found(check(repo, title="fix(core): a")) == []


def test_p8_and_p9_are_warnings(repo: Path) -> None:
    big = "".join(f"pub fn f{i}() {{}}\n" for i in range(1600))
    commit(repo, {"crates/grafeo-core/src/big.rs": big})
    result = check(repo)
    assert (result.returncode, found(result)) == (0, ["P8 PR", "P9 PR"])
    assert "1600 added lines outside tests" in result.stdout


def test_rules_read_the_pull_request_not_the_working_tree(repo: Path) -> None:
    commit(repo, {"docs/guide.md": f"new {EM} text\n"})
    git(repo, "switch", "-q", "release/0.5.44")
    assert found(check(repo)) == ["D2 docs/guide.md:1"]
