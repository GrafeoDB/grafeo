"""Tests for scripts/node-platform-deps.sh: a mistake here ships a package with no native binaries.

A release is the only other place its `add` and `published` run. The registry is a stub `npm` on
the PATH for `published`, and `sleep` is one too, so the retries take no time.

Run: uv run --with pytest python -m pytest scripts/tests/test_node_platform_deps.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "node-platform-deps.sh"
FOLDERS = [
    "darwin-arm64",
    "darwin-x64",
    "linux-arm64-gnu",
    "linux-arm64-musl",
    "linux-x64-gnu",
    "win32-x64-msvc",
]

pytestmark = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("bash", "node", "npm")),
    reason="needs bash, node and npm",
)


def manifest(version: str = "1.2.3", **more) -> dict:
    return {"name": "@grafeo-db/js", "version": version, "main": "index.js", **more}


def package(
    tmp_path: Path, folders: list[str] | None = None, version: str = "1.2.3"
) -> Path:
    """A package as the repository holds it, with its platform folders."""
    root = tmp_path / "node"
    root.mkdir()
    (root / "package.json").write_text(json.dumps(manifest(version)), encoding="utf-8")
    (root / "index.js").write_text("", encoding="utf-8")
    for folder in FOLDERS if folders is None else folders:
        path = root / "npm" / folder
        path.mkdir(parents=True)
        (path / "package.json").write_text(
            json.dumps({"name": f"@grafeo-db/js-{folder}", "version": version}),
            encoding="utf-8",
        )
    return root


def run(
    root: Path, *args: str, env: dict | None = None
) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    return subprocess.run(
        [bash, SCRIPT.as_posix(), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


def listed(root: Path) -> dict:
    return json.loads((root / "package.json").read_text(encoding="utf-8")).get(
        "optionalDependencies", {}
    )


def stubs(tmp_path: Path, answers: list[str]) -> dict:
    """A PATH on which `npm view` gives `answers` in turn, and `sleep` does nothing."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    counter = tmp_path / "calls.txt"
    counter.write_text("0", encoding="utf-8")
    script = f"""#!/usr/bin/env bash
count=$(cat '{counter.as_posix()}')
echo $((count + 1)) > '{counter.as_posix()}'
node -e 'const a = require("{(tmp_path / "answers.json").as_posix()}"); process.stdout.write(a[Math.min('$count', a.length - 1)])'
"""
    (bin_dir / "npm").write_text(script, encoding="utf-8", newline="\n")
    (bin_dir / "sleep").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n"
    )
    for tool in ("npm", "sleep"):
        (bin_dir / tool).chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def published(version: str = "1.2.3", leave_out: int = 0) -> str:
    names = FOLDERS[leave_out:]
    return json.dumps({f"@grafeo-db/js-{name}": version for name in names})


# ------------------------------------------------------------------ usage


@pytest.mark.parametrize(
    "args", [[], ["nonsense"], ["add"], ["published"], ["add", "1", "2"]]
)
def test_a_wrong_call_fails_and_says_how(tmp_path: Path, args: list[str]) -> None:
    result = run(package(tmp_path), *args)
    assert result.returncode == 1
    assert "node-platform-deps:" in result.stderr


# ------------------------------------------------------------------ add


def test_add_lists_every_folder_at_the_version(tmp_path: Path) -> None:
    root = package(tmp_path)
    result = run(root, "add", "1.2.3")
    assert result.returncode == 0, result.stderr
    assert listed(root) == {f"@grafeo-db/js-{name}": "1.2.3" for name in FOLDERS}
    assert "6 platform packages at 1.2.3" in result.stdout


def test_add_refuses_a_manifest_that_lists_them_already(tmp_path: Path) -> None:
    root = package(tmp_path)
    (root / "package.json").write_text(
        json.dumps(
            manifest(optionalDependencies={"@grafeo-db/js-darwin-x64": "1.2.2"})
        ),
        encoding="utf-8",
    )
    result = run(root, "add", "1.2.3")
    assert result.returncode == 1
    assert "lists optional dependencies already" in result.stderr
    assert listed(root) == {"@grafeo-db/js-darwin-x64": "1.2.2"}


def test_add_refuses_a_folder_at_another_version(tmp_path: Path) -> None:
    root = package(tmp_path)
    folder = root / "npm" / "linux-x64-gnu" / "package.json"
    folder.write_text(
        json.dumps({"name": "@grafeo-db/js-linux-x64-gnu", "version": "1.2.2"}),
        encoding="utf-8",
    )
    result = run(root, "add", "1.2.3")
    assert result.returncode == 1
    assert "linux-x64-gnu" in result.stderr
    assert "1.2.2" in result.stderr


@pytest.mark.parametrize("folders", [FOLDERS[:5], [*FOLDERS, "linux-riscv64-gnu"]])
def test_add_refuses_another_number_of_folders(
    tmp_path: Path, folders: list[str]
) -> None:
    root = package(tmp_path, folders)
    result = run(root, "add", "1.2.3")
    assert result.returncode == 1
    assert f"found {len(folders)} folders under npm/, expected 6" in result.stderr


def test_add_with_no_folders_fails(tmp_path: Path) -> None:
    result = run(package(tmp_path, []), "add", "1.2.3")
    assert result.returncode == 1


def test_add_refuses_a_folder_without_a_manifest(tmp_path: Path) -> None:
    root = package(tmp_path)
    (root / "npm" / "linux-x64-gnu" / "package.json").unlink()
    assert run(root, "add", "1.2.3").returncode == 1


# ------------------------------------------------------------------ check


def test_check_passes_and_leaves_the_manifest_alone(tmp_path: Path) -> None:
    root = package(tmp_path)
    before = (root / "package.json").read_bytes()
    result = run(root, "check")
    assert result.returncode == 0, result.stderr
    assert (root / "package.json").read_bytes() == before
    assert listed(root) == {}


def test_check_fails_on_another_number_of_folders(tmp_path: Path) -> None:
    root = package(tmp_path, FOLDERS[:5])
    before = (root / "package.json").read_bytes()
    assert run(root, "check").returncode == 1
    assert (root / "package.json").read_bytes() == before


def test_check_fails_on_a_folder_at_another_version(tmp_path: Path) -> None:
    root = package(tmp_path)
    (root / "npm" / "darwin-x64" / "package.json").write_text(
        json.dumps({"name": "@grafeo-db/js-darwin-x64", "version": "1.2.2"}),
        encoding="utf-8",
    )
    assert run(root, "check").returncode == 1


def test_check_fails_when_the_manifest_lists_them(tmp_path: Path) -> None:
    root = package(tmp_path)
    (root / "package.json").write_text(
        json.dumps(
            manifest(optionalDependencies={"@grafeo-db/js-darwin-x64": "1.2.3"})
        ),
        encoding="utf-8",
    )
    assert run(root, "check").returncode == 1


# ------------------------------------------------------------------ published


def test_published_passes_when_npm_lists_them_all(tmp_path: Path) -> None:
    root = package(tmp_path)
    result = run(root, "published", "1.2.3", env=stubs(tmp_path, [published()]))
    assert result.returncode == 0, result.stderr


def test_published_waits_for_a_package_to_show_up(tmp_path: Path) -> None:
    root = package(tmp_path)
    env = stubs(tmp_path, ["", "", published()])
    result = run(root, "published", "1.2.3", env=env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").strip() == "3"


def test_published_gives_up_when_it_never_shows_up(tmp_path: Path) -> None:
    root = package(tmp_path)
    result = run(root, "published", "1.2.3", env=stubs(tmp_path, [""]))
    assert result.returncode == 1
    assert "lists no optional dependencies on npm" in result.stderr
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").strip() == "6"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(published(leave_out=1), id="one_missing"),
        pytest.param(published(version="1.2.2"), id="another_version"),
        pytest.param(
            json.dumps(
                {**json.loads(published()), "@grafeo-db/js-linux-riscv64-gnu": "1.2.3"}
            ),
            id="one_more",
        ),
    ],
)
def test_published_fails_on_a_wrong_list(tmp_path: Path, answer: str) -> None:
    root = package(tmp_path)
    result = run(root, "published", "1.2.3", env=stubs(tmp_path, [answer]))
    assert result.returncode == 1
    assert "not the platform packages" in result.stderr


# ------------------------------------------------------------------ binaries


def with_binaries(root: Path, sizes: dict[str, int] | None = None) -> None:
    """Gives each folder a `main`, with a binary of the size given (default 3 bytes)."""
    for folder in FOLDERS:
        manifest_path = root / "npm" / folder / "package.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_data["main"] = f"grafeo.{folder}.node"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        size = (sizes or {}).get(folder, 3)
        if size >= 0:
            (root / "npm" / folder / manifest_data["main"]).write_bytes(b"x" * size)


def test_binaries_passes_when_every_folder_has_one(tmp_path: Path) -> None:
    root = package(tmp_path)
    with_binaries(root)
    result = run(root, "binaries")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("size", [-1, 0], ids=["missing", "empty"])
def test_binaries_fails_on_one_missing_or_empty(tmp_path: Path, size: int) -> None:
    root = package(tmp_path)
    with_binaries(root, {"linux-x64-gnu": size})
    result = run(root, "binaries")
    assert result.returncode == 1
    assert "linux-x64-gnu" in result.stderr


def test_binaries_fails_on_a_directory_named_as_the_binary(tmp_path: Path) -> None:
    root = package(tmp_path)
    with_binaries(root, {"linux-x64-gnu": -1})
    directory = root / "npm" / "linux-x64-gnu" / "grafeo.linux-x64-gnu.node"
    directory.mkdir()
    (directory / "a_file").write_bytes(b"x")
    result = run(root, "binaries")
    assert result.returncode == 1
    assert "linux-x64-gnu" in result.stderr


def test_binaries_fails_on_a_folder_with_no_main(tmp_path: Path) -> None:
    root = package(tmp_path)
    assert run(root, "binaries").returncode == 1


def test_binaries_fails_on_another_number_of_folders(tmp_path: Path) -> None:
    root = package(tmp_path, FOLDERS[:5])
    assert run(root, "binaries").returncode == 1


# ------------------------------------------------------------------ the package itself


def test_the_repository_package_passes_the_check() -> None:
    root = ROOT / "crates" / "bindings" / "node"
    before = (root / "package.json").read_bytes()
    result = run(root, "check")
    assert result.returncode == 0, result.stderr
    assert (root / "package.json").read_bytes() == before
