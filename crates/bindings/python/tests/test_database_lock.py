"""A database file has one writer at a time (#405).

Two processes once opened the same file, and the one that closed last
replaced the other's writes. Now a second open of an open file, read-write
or read-only, in this process or another, fails with a "locked" error and
changes nothing; the writer's commits are all there after it closes.
"""

import subprocess
import sys

import pytest
from grafeo import GrafeoDB, GrafeoError

# The reporter's path had no extension; both kinds are checked.
NAMES = ["people", "people.grafeo"]


def names(db):
    return sorted(row["n"] for row in db.execute("MATCH (p:Person) RETURN p.name AS n"))


def open_in_another_process(path, opener):
    """Runs `opener(path)` and an insert in a child process: its result."""
    script = (
        "from grafeo import GrafeoDB\n"
        f"db = {opener}({path!r})\n"
        "db.execute(\"INSERT (:Person {name: 'X'})\")\n"
        "db.close()\n"
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("name", NAMES)
def test_a_second_open_in_this_process_is_refused(tmp_path, name):
    path = str(tmp_path / name)
    first = GrafeoDB(path)
    first.execute("INSERT (:Person {name: 'A'})")
    with pytest.raises(GrafeoError, match="locked"):
        GrafeoDB(path)
    with pytest.raises(GrafeoError, match="locked"):
        GrafeoDB.open_read_only(path)
    first.execute("INSERT (:Person {name: 'B'})")
    first.close()

    db = GrafeoDB(path)
    assert names(db) == ["A", "B"]
    db.close()


@pytest.mark.parametrize("name", NAMES)
def test_another_process_cannot_open_it_and_writes_nothing(tmp_path, name):
    path = str(tmp_path / name)
    first = GrafeoDB(path)
    first.execute("INSERT (:Person {name: 'A'})")
    for opener in ("GrafeoDB", "GrafeoDB.open_read_only"):
        child = open_in_another_process(path, opener)
        assert child.returncode != 0, f"{opener} opened a locked file"
        assert "locked" in child.stderr, child.stderr
    first.execute("INSERT (:Person {name: 'B'})")
    first.close()

    db = GrafeoDB(path)
    assert names(db) == ["A", "B"], "the writer's commits, and nothing of the refused opens"
    db.close()


def test_once_closed_another_process_opens_it(tmp_path):
    path = str(tmp_path / "people.grafeo")
    first = GrafeoDB(path)
    first.execute("INSERT (:Person {name: 'A'})")
    first.close()
    child = open_in_another_process(path, "GrafeoDB")
    assert child.returncode == 0, child.stderr
    db = GrafeoDB(path)
    assert names(db) == ["A", "X"]
    db.close()


def test_read_only_opens_share_the_file(tmp_path):
    path = str(tmp_path / "people.grafeo")
    db = GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'A'})")
    db.close()
    first = GrafeoDB.open_read_only(path)
    second = GrafeoDB.open_read_only(path)
    assert names(first) == names(second) == ["A"]
    with pytest.raises(GrafeoError, match="locked"):
        GrafeoDB(path)
    first.close()
    second.close()
