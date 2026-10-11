"""A call to an unknown function raises GrafeoError, and GQL has `=~`.

Reported downstream (Deriva, 2026-10-09), on 0.5.44 and the 0.6.0
development wheel: `no_such_function(x)` and a typo such as `upperr(x)` were
null for every row, in GQL and Cypher, so a WHERE with a misspelled function
kept no rows and said nothing. Such a call now raises an error before any
row is read, with the closest function name as a hint.

GQL refused `n.path =~ '...'` ("Expected expression"), which Deriva's
derivation queries use to leave out test, cache and version control paths.
GQL now has `=~` with the meaning of Cypher's (the whole string must match),
and a pattern that is not a regular expression raises an error that names it.
"""

import grafeo
import pytest

HAS_CYPHER = hasattr(grafeo.GrafeoDB(), "execute_cypher")

LANGUAGES = [
    pytest.param(False, id="gql"),
    pytest.param(
        True,
        id="cypher",
        marks=pytest.mark.skipif(not HAS_CYPHER, reason="grafeo built without cypher feature"),
    ),
]

EXCLUDED = r"'.*(test|spec|__pycache__|node_modules|\.git).*'"


def run(db, query, cypher):
    return db.execute_cypher(query) if cypher else db.execute(query)


@pytest.fixture
def db():
    database = grafeo.GrafeoDB()
    database.execute(
        "INSERT (:Directory {path: 'src/tests'}), (:Directory {path: 'src/app'}), "
        "(:Directory {path: 'src/.git/hooks'}), (:Directory {path: 'src/agit'})"
    )
    return database


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_an_unknown_function_raises(db, cypher):
    with pytest.raises(grafeo.GrafeoError) as raised:
        run(db, "MATCH (d:Directory) RETURN no_such_function(d.path) AS x", cypher)
    assert "Unknown function 'no_such_function'" in str(raised.value)


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_a_misspelled_function_names_the_close_one(db, cypher):
    query = "MATCH (d:Directory) WHERE upperr(d.path) = 'SRC/APP' RETURN d.path AS path"
    with pytest.raises(grafeo.GrafeoError) as raised:
        run(db, query, cypher)
    message = str(raised.value)
    assert "Unknown function 'upperr'" in message
    assert "Did you mean 'upper'?" in message


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_regex_match_leaves_out_the_excluded_paths(db, cypher):
    query = f"MATCH (d:Directory) WHERE NOT d.path =~ {EXCLUDED} RETURN d.path AS path"
    paths = sorted(row["path"] for row in run(db, query, cypher))
    assert paths == ["src/agit", "src/app"]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_an_invalid_pattern_raises(db, cypher):
    with pytest.raises(grafeo.GrafeoError) as raised:
        run(db, "MATCH (d:Directory) WHERE d.path =~ '(src' RETURN d.path", cypher)
    assert "Invalid regular expression '(src'" in str(raised.value)
