"""The path search budget from Python (#492): `GrafeoDB(path_search_budget=...)`. Zero and
negative values are refused; the budget binds, with its error code and the keyword named in the
message. The snapshot retention limit, `GrafeoDB(snapshot_retention_limit=...)`: below 1 MiB is
refused; past it an open transaction raises GRAFEO-T012.
"""

import pytest

grafeo = pytest.importorskip("grafeo")

STAR = (
    "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), "
    "(c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})"
)
PATHS = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n"


def star(**keywords):
    db = grafeo.GrafeoDB(**keywords)
    db.execute(STAR)
    return db


@pytest.mark.parametrize("value", [0, -1])
def test_zero_and_negative_budgets_are_refused(value):
    with pytest.raises((ValueError, OverflowError, grafeo.GrafeoError)):
        grafeo.GrafeoDB(path_search_budget=value)


def test_the_default_runs_the_paths():
    assert len(list(star().execute(PATHS))) == 5


def test_the_path_search_budget_binds():
    db = star(path_search_budget=1)
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute(PATHS)
    assert raised.value.error_code == "GRAFEO-S001", str(raised.value)
    assert "path_search_budget" in str(raised.value), str(raised.value)


def test_a_snapshot_retention_limit_below_1_mib_is_refused():
    with pytest.raises(grafeo.GrafeoError, match="at least 1 MiB"):
        grafeo.GrafeoDB(snapshot_retention_limit=512 * 1024)


def test_the_snapshot_retention_limit_binds():
    """Past the limit (20 commits replace 64 KiB each, over 1 MiB) the open transaction that read
    Alix raises GRAFEO-T012, and rolls back."""
    db = grafeo.GrafeoDB(snapshot_retention_limit=1 << 20)
    db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
    read_bio = "MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio"
    tx = db.begin_transaction()
    tx.execute(read_bio)
    text = "Vincent " * (8 * 1024)
    for round_ in range(20):
        db.execute(f"MATCH (p:Person {{name: 'Alix'}}) SET p.bio = '{round_}{text}'")
    with pytest.raises(grafeo.GrafeoError) as raised:
        tx.execute(read_bio)
    assert raised.value.error_code == "GRAFEO-T012", str(raised.value)
    tx.rollback()


def test_a_zero_history_retention_limit_is_refused():
    with pytest.raises(grafeo.GrafeoError, match="history_retention_limit"):
        grafeo.GrafeoDB(history_retention_limit=0)


def test_the_history_retention_limit_moves_the_start_of_the_window():
    """Forty commits replacing 64 KiB each pass a 1 MiB limit: the first epoch is no longer
    retained, and the latest reads."""
    db = grafeo.GrafeoDB(history_retention="forever", history_retention_limit=1 << 20)
    db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
    first = db.current_epoch()
    bio = "V" * (64 * 1024)
    for round_ in range(40):
        db.execute(f"MATCH (p:Person {{name: 'Alix'}}) SET p.bio = '{round_:04}{bio}'")
    db.gc()
    with pytest.raises(grafeo.GrafeoError, match="not retained"):
        db.as_of(first)
