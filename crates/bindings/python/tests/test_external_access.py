"""A session without external access reads and writes nothing outside the database.

`db.session(read_only=False, external_access=True)` gives a `grafeo.Session`. With
`external_access=False` every statement on it that reaches outside (LOAD DATA in each form,
Cypher LOAD CSV, the load procedures, SPARQL LOAD) is refused at planning, before any file is
opened, EXPLAIN included: the permission-denied error (GQLSTATUS class 42), naming external
access. The bound holds for every statement on the session: auto-commit, its transactions, every
language method and the lazy stream. READ ONLY and external access are independent, the defaults
are today's `db.execute`, and the application's own calls (`db.load_nodes_csv`) are unaffected.

The oracle: the same statement on a default session answers as `db.execute` does today, and a
refused one changes nothing. A file that does not exist tells the refusal comes before the open:
a session that opened it would say it cannot find it.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

grafeo = pytest.importorskip("grafeo")


@dataclass
class Files:
    csv: str
    bare_csv: str
    jsonl: str
    turtle: str
    missing: str


@pytest.fixture
def files(tmp_path: Path) -> Files:
    (tmp_path / "people.csv").write_text("id,name\n1,Alix\n2,Gus\n", encoding="utf-8")
    (tmp_path / "bare.csv").write_text("1,Alix\n2,Gus\n", encoding="utf-8")
    (tmp_path / "people.jsonl").write_text(
        '{"id": 1, "name": "Alix"}\n{"id": 2, "name": "Gus"}\n', encoding="utf-8"
    )
    (tmp_path / "people.ttl").write_text("<urn:alix> <urn:knows> <urn:gus> .\n", encoding="utf-8")
    return Files(
        csv=(tmp_path / "people.csv").as_posix(),
        bare_csv=(tmp_path / "bare.csv").as_posix(),
        jsonl=(tmp_path / "people.jsonl").as_posix(),
        turtle=(tmp_path / "people.ttl").as_posix(),
        missing=(tmp_path / "nobody-wrote-this.csv").as_posix(),
    )


def load_data(path: str) -> str:
    return f"LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name"


def load_procedure(path: str) -> str:
    return f"CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Person'], key: 'id'}})"


def file_url(path: str) -> str:
    return "file:///" + path.lstrip("/")


# Each form: an id, the session method, the query from the files, whether a default session
# answers it, and the features it needs. SPARQL LOAD is refused as not supported yet even with
# access, so only its refusal is checked.
FORMS = [
    ("gql-load-data", "execute", lambda f: load_data(f.csv), True, []),
    (
        "gql-load-data-without-headers",
        "execute",
        lambda f: f"LOAD DATA FROM '{f.bare_csv}' FORMAT CSV AS row RETURN row[1] AS name",
        True,
        [],
    ),
    (
        "gql-load-data-file-url",
        "execute",
        lambda f: (
            f"LOAD DATA FROM '{file_url(f.csv)}' FORMAT CSV WITH HEADERS AS row "
            "RETURN row.name AS name"
        ),
        True,
        [],
    ),
    ("gql-load-data-missing-file", "execute", lambda f: load_data(f.missing), False, []),
    ("gql-load-nodes-csv", "execute", lambda f: load_procedure(f.csv), True, []),
    (
        "gql-load-nodes-csv-then-match",
        "execute",
        lambda f: (
            load_procedure(f.csv)
            + " YIELD rows RETURN rows NEXT MATCH (p:Person) RETURN count(p) AS people"
        ),
        True,
        [],
    ),
    # With access the next check is the arguments', so this one only shows the access check
    # comes first.
    (
        "gql-explain-load-nodes-csv-without-arguments",
        "execute",
        lambda f: "EXPLAIN CALL grafeo.load_nodes_csv()",
        False,
        [],
    ),
    ("gql-explain-load-data", "execute", lambda f: "EXPLAIN " + load_data(f.csv), True, []),
    ("gql-profile-load-data", "execute", lambda f: "PROFILE " + load_data(f.csv), True, []),
    ("gql-lazy-load-data", "execute_lazy", lambda f: load_data(f.csv), True, []),
    (
        "gql-through-execute-language",
        "execute_language:gql",
        lambda f: load_data(f.csv),
        True,
        [],
    ),
    (
        "cypher-load-csv",
        "execute_cypher",
        lambda f: f"LOAD CSV WITH HEADERS FROM '{file_url(f.csv)}' AS row RETURN row.name AS name",
        True,
        ["cypher"],
    ),
    (
        "cypher-load-csv-bare-path",
        "execute_cypher",
        lambda f: f"LOAD CSV FROM '{f.bare_csv}' AS row RETURN row[1] AS name",
        True,
        ["cypher"],
    ),
    (
        "cypher-load-nodes-csv-then-match",
        "execute_cypher",
        lambda f: load_procedure(f.csv) + " YIELD rows MATCH (p:Person) RETURN p.name AS name",
        True,
        ["cypher"],
    ),
    ("sql-load-nodes-csv", "execute_sql", lambda f: load_procedure(f.csv), True, ["sql-pgq"]),
    (
        "gql-load-data-jsonl",
        "execute",
        lambda f: f"LOAD DATA FROM '{f.jsonl}' FORMAT JSONL AS row RETURN row.name AS name",
        True,
        ["jsonl-import"],
    ),
    (
        "gql-load-data-parquet-missing-file",
        "execute",
        lambda f: f"LOAD DATA FROM '{f.missing}.parquet' FORMAT PARQUET AS row RETURN row",
        False,
        ["parquet-import"],
    ),
    (
        "sparql-load",
        "execute_sparql",
        lambda f: f"LOAD <{file_url(f.turtle)}>",
        False,
        ["sparql"],
    ),
    (
        "sparql-load-into-graph",
        "execute_sparql",
        lambda f: f"LOAD <{file_url(f.turtle)}> INTO GRAPH <urn:people>",
        False,
        ["sparql"],
    ),
]


def form_params(only_running: bool = False):
    return [
        pytest.param(method, query, id=name, marks=[pytest.mark.requires(*needs)] if needs else [])
        for name, method, query, runs, needs in FORMS
        if runs or not only_running
    ]


def call(target, method: str, query: str):
    """Runs `query` through `method` of a session, transaction or database, and reads every row,
    so a lazy stream runs too."""
    if method.startswith("execute_language:"):
        result = target.execute_language(method.split(":", 1)[1], query)
    else:
        result = getattr(target, method)(query)
    return list(result)


def assert_refused(run):
    """`run` raises the external-access refusal: a GrafeoError of class 42 naming it."""
    with pytest.raises(grafeo.GrafeoError) as raised:
        run()
    error = raised.value
    assert "external access" in str(error), str(error)
    assert error.gql_status.startswith("42"), (error.gql_status, str(error))


def people(db) -> int:
    return len(list(db.execute("MATCH (p:Person) RETURN p")))


# ---------------------------------------------------------------------------
# The forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "query"), form_params())
def test_every_form_is_refused_without_external_access(files, method, query):
    db = grafeo.GrafeoDB()
    session = db.session(external_access=False)
    assert_refused(lambda: call(session, method, query(files)))
    assert people(db) == 0, "a refused load changed the graph"


@pytest.mark.parametrize(("method", "query"), form_params(only_running=True))
def test_every_form_runs_on_a_default_session(files, method, query):
    # A database per statement, so one load does not meet another's nodes.
    call(grafeo.GrafeoDB().session(), method, query(files))
    call(grafeo.GrafeoDB().session(external_access=True), method, query(files))
    # A load procedure also writes: a read-only session refuses it as a write, which the order
    # of refusals test pins.
    if not query(files).startswith("CALL"):
        call(grafeo.GrafeoDB().session(read_only=True), method, query(files))
    call(grafeo.GrafeoDB(), method, query(files))


def test_a_missing_file_fails_differently_with_access(files):
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError) as raised:
        call(db.session(), "execute", load_data(files.missing))
    assert "external access" not in str(raised.value)
    assert not raised.value.gql_status.startswith("42"), str(raised.value)
    assert_refused(
        lambda: call(db.session(external_access=False), "execute", load_data(files.missing))
    )


# ---------------------------------------------------------------------------
# The options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "writes", "loads"),
    [
        ({}, True, True),
        ({"read_only": False, "external_access": True}, True, True),
        ({"read_only": True}, False, True),
        ({"external_access": False}, True, False),
        ({"read_only": True, "external_access": False}, False, False),
    ],
    ids=["defaults", "explicit-defaults", "read-only", "no-external-access", "both"],
)
def test_read_only_and_external_access_are_independent(files, options, writes, loads):
    db = grafeo.GrafeoDB()
    session = db.session(**options)
    write = "INSERT (:Person {name: 'Mia'})"
    if writes:
        session.execute(write)
        assert people(db) == 1
    else:
        with pytest.raises(grafeo.GrafeoError) as raised:
            session.execute(write)
        assert "external access" not in str(raised.value), str(raised.value)
        assert people(db) == 0
    if loads:
        assert [row["name"] for row in session.execute(load_data(files.csv))] == ["Alix", "Gus"]
    else:
        assert_refused(lambda: session.execute(load_data(files.csv)))


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {"external": False}),
        ((), {"readOnly": True}),
        ((), {"externalAccess": False}),
        ((True,), {}),
        ((False, False), {}),
    ],
    ids=["external", "readOnly", "externalAccess", "one-positional", "two-positional"],
)
def test_an_unknown_or_positional_option_is_refused(args, kwargs):
    with pytest.raises(TypeError):
        grafeo.GrafeoDB().session(*args, **kwargs)


# ---------------------------------------------------------------------------
# Every statement on the session is bound
# ---------------------------------------------------------------------------


@pytest.mark.requires("cypher")
def test_the_transactions_of_the_session_are_bound(files):
    db = grafeo.GrafeoDB()
    session = db.session(external_access=False)
    tx = session.begin_transaction()
    assert_refused(lambda: tx.execute(load_data(files.csv)))
    assert_refused(lambda: tx.execute(load_procedure(files.csv)))
    assert_refused(
        lambda: tx.execute_cypher(
            f"LOAD CSV WITH HEADERS FROM '{file_url(files.csv)}' AS row RETURN row.name AS name"
        )
    )
    tx.execute("INSERT (:Person {name: 'Mia'})")
    tx.rollback()
    assert_refused(lambda: session.execute(load_data(files.csv)))
    with session.begin_transaction("read_committed") as tx:
        assert_refused(lambda: tx.execute(load_data(files.csv)))
    assert people(db) == 0


def test_a_cached_plan_is_checked_again(files):
    db = grafeo.GrafeoDB()
    query = "CALL grafeo.load_nodes_csv({path: $path, labels: ['Person'], key: 'id'})"
    db.execute(query, {"path": files.csv})
    assert people(db) == 2
    restricted = db.session(external_access=False)
    for _ in range(2):
        assert_refused(lambda: restricted.execute(query, {"path": files.csv}))
    assert people(db) == 2


def test_a_composed_statement_runs_nothing(files):
    db = grafeo.GrafeoDB()
    session = db.session(external_access=False)
    assert_refused(
        lambda: session.execute(f"INSERT (:Marker) RETURN 1 AS one NEXT {load_data(files.csv)}")
    )
    assert len(list(db.execute("MATCH (m:Marker) RETURN m"))) == 0


def test_no_statement_gives_the_session_access(files):
    session = grafeo.GrafeoDB().session(external_access=False)
    session.execute("SESSION RESET")
    assert_refused(lambda: session.execute(load_data(files.csv)))


@pytest.mark.parametrize(
    ("method", "query"),
    [
        pytest.param("execute", "MATCH (n) RETURN count(n) AS n", id="gql"),
        pytest.param(
            "execute_cypher",
            "MATCH (n) RETURN count(n) AS n",
            id="cypher",
            marks=pytest.mark.requires("cypher"),
        ),
        pytest.param(
            "execute_sql",
            "SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.name AS name))",
            id="sql",
            marks=pytest.mark.requires("sql-pgq"),
        ),
        pytest.param(
            "execute_gremlin", "g.V().count()", id="gremlin", marks=pytest.mark.requires("gremlin")
        ),
        pytest.param(
            "execute_graphql",
            "{ Person { name } }",
            id="graphql",
            marks=pytest.mark.requires("graphql"),
        ),
        pytest.param(
            "execute_sparql",
            "SELECT ?s WHERE { ?s ?p ?o }",
            id="sparql",
            marks=pytest.mark.requires("sparql"),
        ),
        pytest.param("execute_lazy", "MATCH (n:Person) RETURN n.name", id="lazy"),
        pytest.param(
            "execute_language:gql", "CALL grafeo.procedures() YIELD name RETURN name", id="language"
        ),
    ],
)
def test_everything_inside_the_database_still_runs(method, query):
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})")
    call(db.session(external_access=False), method, query)


# ---------------------------------------------------------------------------
# The application's own calls, and the session's lifetime
# ---------------------------------------------------------------------------


def test_the_applications_own_calls_are_not_bound(files):
    db = grafeo.GrafeoDB()
    restricted = db.session(external_access=False)
    assert db.load_nodes_csv(files.csv, ["Person"], key="id")["created"] == 2
    assert [row["name"] for row in db.execute(load_data(files.csv))] == ["Alix", "Gus"]
    assert_refused(lambda: restricted.execute(load_data(files.csv)))


def test_a_closed_session_refuses_statements_and_others_go_on(files):
    db = grafeo.GrafeoDB()
    closed = db.session()
    other = db.session(external_access=False)
    closed.close()
    with pytest.raises(grafeo.GrafeoError):
        closed.execute("RETURN 1 AS one")
    with pytest.raises(grafeo.GrafeoError):
        closed.begin_transaction()
    assert [row["one"] for row in other.execute("RETURN 1 AS one")] == [1]
    assert_refused(lambda: other.execute(load_data(files.csv)))
    assert [row["one"] for row in db.execute("RETURN 1 AS one")] == [1]


def test_the_with_block_closes_the_session(files):
    db = grafeo.GrafeoDB()
    with db.session(external_access=False) as session:
        assert isinstance(session, grafeo.Session)
        assert_refused(lambda: session.execute(load_data(files.csv)))
    with pytest.raises(grafeo.GrafeoError):
        session.execute("RETURN 1 AS one")


# ---------------------------------------------------------------------------
# Deriva's case
# ---------------------------------------------------------------------------


@pytest.mark.requires("cypher")
def test_an_untrusted_session_reads_no_file(tmp_path):
    """Deriva's studio shares one open database with a writing pipeline and runs user queries on
    an untrusted session. On 0.6.1 both probe queries, inside a transaction rolled back, returned
    the secret file's rows. The load procedures also write, and the external-access refusal comes
    before the write refusal, so all three name it."""
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Pipeline {name: 'Butch'})")
    secret = tmp_path / "secret.csv"
    secret.write_text("name\nHans\nShosanna\n", encoding="utf-8")
    path = secret.as_posix()
    session = db.session(read_only=True, external_access=False)

    tx = session.begin_transaction()
    for run in [
        lambda: tx.execute_cypher(
            f"CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Leak']}}) "
            "YIELD rows MATCH (n:Leak) RETURN n.name"
        ),
        lambda: tx.execute(f"CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Leak']}})"),
        lambda: tx.execute_cypher(
            f"LOAD CSV WITH HEADERS FROM '{file_url(path)}' AS row RETURN row.name"
        ),
    ]:
        assert_refused(run)
    tx.rollback()

    assert [row["n"] for row in session.execute("MATCH (n) RETURN count(n) AS n")] == [1]
    assert len(list(db.execute("MATCH (n:Leak) RETURN n"))) == 0


@pytest.mark.parametrize(
    ("options", "access_named"),
    [
        ({"read_only": True, "external_access": False}, True),
        ({"read_only": True}, False),
    ],
    ids=["read-only-without-access", "read-only-with-access"],
)
def test_external_access_is_refused_before_the_write(files, options, access_named):
    """A load procedure reaches outside and writes: a read-only session without access is told
    about external access; once access is granted, the write is what it refuses."""
    db = grafeo.GrafeoDB()
    session = db.session(**options)
    if access_named:
        assert_refused(lambda: session.execute(load_procedure(files.csv)))
    else:
        with pytest.raises(grafeo.GrafeoError) as raised:
            session.execute(load_procedure(files.csv))
        assert "external access" not in str(raised.value), str(raised.value)
    assert people(db) == 0
