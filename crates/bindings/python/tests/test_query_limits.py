"""The path search budget from Python (#492): `GrafeoDB(path_search_budget=...)`. Zero and
negative values are refused; the budget binds, with its error code and the keyword named in the
message.
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
