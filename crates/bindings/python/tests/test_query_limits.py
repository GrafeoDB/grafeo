"""The query resource limits from Python (#492): `GrafeoDB(path_search_budget=...,
path_expansion_limit=..., max_query_length=..., max_identifier_length=...)`. Zero and negative
values are refused; each setting binds, with its error code and the keyword named in the message.
The snapshot retention limit, `GrafeoDB(snapshot_retention_limit=...)`: below 1 MiB is refused;
past it an open transaction raises GRAFEO-T012.
"""

import pytest

grafeo = pytest.importorskip("grafeo")

STAR = (
    "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), "
    "(c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})"
)
PATHS = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n"
KEYWORDS = [
    "path_search_budget",
    "path_expansion_limit",
    "max_query_length",
    "max_identifier_length",
]


def star(**keywords):
    db = grafeo.GrafeoDB(**keywords)
    db.execute(STAR)
    return db


def refused(db, query):
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute(query)
    return raised.value


@pytest.mark.parametrize("keyword", KEYWORDS)
@pytest.mark.parametrize("value", [0, -1])
def test_zero_and_negative_values_are_refused(keyword, value):
    with pytest.raises((ValueError, OverflowError, grafeo.GrafeoError)):
        grafeo.GrafeoDB(**{keyword: value})


def test_the_defaults_run_the_statements():
    assert len(list(star().execute(PATHS))) == 5


def test_the_path_expansion_limit_binds():
    error = refused(star(path_expansion_limit=4), PATHS)
    assert error.error_code == "GRAFEO-Q009", str(error)
    assert "path_expansion_limit" in str(error), str(error)
    assert len(list(star(path_expansion_limit=5).execute(PATHS))) == 5


def test_the_path_search_budget_binds():
    error = refused(star(path_search_budget=1), PATHS)
    assert error.error_code == "GRAFEO-S001", str(error)
    assert "path_search_budget" in str(error), str(error)


def test_the_query_length_binds():
    db = star(max_query_length=200)
    assert len(list(db.execute(PATHS))) == 5
    error = refused(db, PATHS + " " * 200)
    assert error.error_code == "GRAFEO-Q002", str(error)
    assert "max_query_length" in str(error), str(error)


def test_the_identifier_length_binds():
    db = star(max_identifier_length=8)
    assert len(list(db.execute("MATCH (abcdefgh:Star) RETURN abcdefgh.n"))) == 1
    error = refused(db, "MATCH (abcdefghi:Star) RETURN 1")
    assert error.error_code == "GRAFEO-Q002", str(error)
    assert "max_identifier_length" in str(error), str(error)


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
