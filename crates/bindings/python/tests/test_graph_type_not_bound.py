"""A graph type's rules hold only in its graphs (#730), and an upgrade must not lose them silently.

Tests first, from the user's decision: `db.validate()` lists the warning with the code
`GRAPH_TYPE_NOT_BOUND`, and Python also raises `warnings.warn` at open, once per open, which a
filter can silence, and `validate()` raises it too while the state holds, keeping its list of
errors. A warning for node and edge types, when a listed type's label has elements in a
graph not bound to it; none for a bound graph, or when there are no such elements.
"""

import warnings
from pathlib import Path

import grafeo
import pytest

CODE = "GRAPH_TYPE_NOT_BOUND"


def write(path: Path, *statements: str) -> None:
    db = grafeo.GrafeoDB(str(path))
    for statement in statements:
        db.execute(statement)
    db.close()


def about_rules(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if CODE in str(w.message)]


def opened(path: Path) -> tuple[grafeo.GrafeoDB, list[str]]:
    """The database at `path`, and the warnings of opening it, with every warning shown."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db = grafeo.GrafeoDB(str(path))
    return db, about_rules(caught)


@pytest.fixture
def in_default_graph(tmp_path: Path) -> Path:
    path = tmp_path / "default.grafeo"
    # Declared by the graph type: a standalone type it lists stays in force (#764).
    write(
        path,
        "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING NOT NULL))",
        "INSERT (:City {size: 1})",
    )
    return path


def test_opening_warns_of_a_listed_label_in_the_default_graph(in_default_graph: Path):
    db, found = opened(in_default_graph)
    db.close()
    assert len(found) == 1, found
    for part in ("City", "atlas", "default"):
        assert part in found[0], found


def test_each_open_warns_once(in_default_graph: Path):
    for _ in range(2):
        db, found = opened(in_default_graph)
        db.close()
        assert len(found) == 1, found


def test_statements_after_the_open_do_not_warn_again(in_default_graph: Path):
    db, _ = opened(in_default_graph)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db.execute("INSERT (:City {size: 2})")
        db.execute("MATCH (c:City) RETURN c.size")
    db.close()
    assert about_rules(caught) == []


def test_a_filter_on_the_code_silences_it(in_default_graph: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=f".*{CODE}.*")
        db = grafeo.GrafeoDB(str(in_default_graph))
    db.close()
    assert about_rules(caught) == []


def test_a_filter_can_turn_it_into_an_error(in_default_graph: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{CODE}.*")
        with pytest.raises(Warning, match=CODE):
            grafeo.GrafeoDB(str(in_default_graph))


def test_it_is_a_warning_category(in_default_graph: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db = grafeo.GrafeoDB(str(in_default_graph))
    db.close()
    found = [w for w in caught if CODE in str(w.message)]
    assert len(found) == 1
    assert issubclass(found[0].category, Warning)


def test_an_edge_type_warns_too(tmp_path: Path):
    path = tmp_path / "edges.grafeo"
    write(
        path,
        "CREATE GRAPH TYPE atlas (EDGE TYPE ROAD (km INT64))",
        "INSERT (:A)-[:ROAD {km: 'far'}]->(:B)",
    )
    db, found = opened(path)
    db.close()
    assert len(found) == 1, found
    assert "ROAD" in found[0], found


def test_a_named_graph_not_bound_warns_and_says_which(tmp_path: Path):
    path = tmp_path / "plain.grafeo"
    write(
        path,
        "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
        "CREATE GRAPH plain",
        "USE plain INSERT (:City {size: 1})",
    )
    db, found = opened(path)
    db.close()
    assert len(found) == 1, found
    assert "plain" in found[0], found


@pytest.mark.parametrize(
    "statements",
    [
        pytest.param(
            (
                "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
                "CREATE GRAPH maps TYPED atlas",
                "USE maps INSERT (:City {name: 'Paris'})",
            ),
            id="only_in_the_bound_graph",
        ),
        pytest.param(
            (
                "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
                "INSERT (:Town {name: 'Paris'})",
            ),
            id="no_nodes_of_the_label",
        ),
        pytest.param(
            (
                "CREATE NODE TYPE City (name STRING NOT NULL)",
                "INSERT (:City {name: 'Paris'})",
            ),
            id="a_type_no_graph_type_lists",
        ),
        pytest.param(
            (
                "CREATE NODE TYPE City (name STRING NOT NULL)",
                "CREATE GRAPH TYPE atlas (NODE TYPE City)",
                "INSERT (:City {name: 'Paris'})",
            ),
            id="a_standalone_type_a_graph_type_lists",
        ),
    ],
)
def test_no_warning_where_no_rule_is_lost(tmp_path: Path, statements: tuple[str, ...]):
    path = tmp_path / "quiet.grafeo"
    write(path, *statements)
    db, found = opened(path)
    db.close()
    assert found == []


def test_a_new_database_does_not_warn(tmp_path: Path):
    db, found = opened(tmp_path / "new.grafeo")
    db.close()
    assert found == []


@pytest.mark.parametrize(
    "open_it",
    [
        pytest.param(grafeo.GrafeoDB, id="constructor"),
        pytest.param(grafeo.GrafeoDB.open, id="open"),
        pytest.param(grafeo.GrafeoDB.open_read_only, id="read_only"),
        pytest.param(grafeo.GrafeoDB.open_in_memory, id="in_memory"),
    ],
)
def test_every_way_to_open_warns_once(in_default_graph: Path, open_it):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db = open_it(str(in_default_graph))
    found = about_rules(caught)
    assert len(found) == 1, found
    assert found[0].startswith(CODE), found
    del db


def test_one_warning_for_each_listed_type(tmp_path: Path):
    path = tmp_path / "two.grafeo"
    write(
        path,
        "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING), NODE TYPE Town (name STRING))",
        "INSERT (:City {size: 1}), (:Town {size: 2})",
    )
    db, found = opened(path)
    db.close()
    assert len(found) == 2, found
    assert sum("City" in w for w in found) == 1, found
    assert sum("Town" in w for w in found) == 1, found


def test_validate_warns_while_the_state_holds_and_keeps_its_list(tmp_path: Path):
    db = grafeo.GrafeoDB()
    db.execute("CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))")
    db.execute("INSERT (:City {size: 1})-[:NEAR]->(:Town {size: 2})")
    with pytest.warns(UserWarning, match=CODE):
        assert db.validate() == []

    db.execute("MATCH (c:City) DETACH DELETE c")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert db.validate() == []
    assert about_rules(caught) == [], "the nodes are deleted"

    db.execute("INSERT (:City {size: 3})-[:NEAR]->(:Town {size: 4})")
    with pytest.warns(UserWarning, match=CODE):
        db.validate()

    db.execute("DROP GRAPH TYPE atlas")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert db.validate() == []
    assert about_rules(caught) == [], "the graph type is dropped"


def test_validate_does_not_raise_other_warnings(tmp_path: Path):
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:City {size: 1})")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert db.validate() == []
    assert caught == [], "NO_EDGES stays out of it"


def test_a_filter_silences_validate_too(tmp_path: Path):
    db = grafeo.GrafeoDB()
    db.execute("CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))")
    db.execute("INSERT (:City {size: 1})")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=f".*{CODE}.*")
        db.validate()
    assert about_rules(caught) == []


def test_pytest_sees_the_warning_at_open(in_default_graph: Path):
    with pytest.warns(UserWarning, match=CODE):
        db = grafeo.GrafeoDB(str(in_default_graph))
    with pytest.warns(UserWarning, match=CODE):
        errors = db.validate()
    assert errors == [], "validate() returns only errors"
    db.close()
