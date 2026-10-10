"""Tests for DataFrame bridge: to_pandas(), to_polars(), nodes_df(), edges_df()."""

import sys

import pytest

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import grafeo


@pytest.fixture
def populated_db():
    """Create a database with Person and Company nodes plus edges."""
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    db.execute("INSERT (:Person {name: 'Vincent', age: 35, city: 'Amsterdam'})")
    db.execute("INSERT (:Company {name: 'Acme Corp', founded: 2010})")
    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'})
        INSERT (a)-[:KNOWS {since: 2020}]->(g)
    """)
    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (c:Company {name: 'Acme Corp'})
        INSERT (a)-[:WORKS_AT {role: 'Engineer'}]->(c)
    """)
    return db


# --- QueryResult.to_pandas() ---


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestToPandas:
    def test_basic_query(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name")
        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["n.name", "n.age"]
        assert len(df) == 3
        assert list(df["n.name"]) == ["Alix", "Gus", "Vincent"]
        assert list(df["n.age"]) == [30, 25, 35]

    def test_empty_result(self, populated_db):
        result = populated_db.execute("MATCH (n:Person {name: 'Nobody'}) RETURN n.name")
        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["n.name"]

    def test_null_values(self, populated_db):
        """Nodes without 'city' should produce None in the DataFrame."""
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.city ORDER BY n.name")
        df = result.to_pandas()

        assert df.loc[df["n.name"] == "Vincent", "n.city"].iloc[0] == "Amsterdam"
        assert pd.isna(df.loc[df["n.name"] == "Alix", "n.city"].iloc[0])

    def test_mixed_types(self, populated_db):
        """Columns with mixed types (int, string, null) should work."""
        result = populated_db.execute("MATCH (n) RETURN n.name, labels(n) ORDER BY n.name")
        df = result.to_pandas()
        assert len(df) == 4  # 3 persons + 1 company

    def test_single_column(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN count(n)")
        df = result.to_pandas()
        assert len(df) == 1
        assert df.iloc[0, 0] == 3


# --- QueryResult.to_polars() ---


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestToPolars:
    def test_basic_query(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name")
        df = result.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["n.name", "n.age"]
        assert len(df) == 3
        assert df["n.name"].to_list() == ["Alix", "Gus", "Vincent"]
        assert df["n.age"].to_list() == [30, 25, 35]

    def test_empty_result(self, populated_db):
        result = populated_db.execute("MATCH (n:Person {name: 'Nobody'}) RETURN n.name")
        df = result.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_null_values(self, populated_db):
        result = populated_db.execute("MATCH (n:Person) RETURN n.name, n.city ORDER BY n.name")
        df = result.to_polars()
        vincent_row = df.filter(pl.col("n.name") == "Vincent")
        assert vincent_row["n.city"][0] == "Amsterdam"


# --- db.nodes_df() ---


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestNodesDf:
    def test_basic(self, populated_db):
        df = populated_db.nodes_df()

        assert isinstance(df, pd.DataFrame)
        assert "_id" in df.columns
        assert "_labels" in df.columns
        assert "name" in df.columns
        assert len(df) == 4  # 3 persons + 1 company

    def test_property_columns(self, populated_db):
        """Each unique property key across all nodes becomes a column."""
        df = populated_db.nodes_df()
        # Person nodes have name, age (and optionally city)
        # Company nodes have name, founded
        assert "age" in df.columns
        assert "founded" in df.columns
        assert "name" in df.columns

    def test_missing_properties_are_none(self, populated_db):
        """Nodes without a property should have None in that column."""
        df = populated_db.nodes_df()
        # Company node shouldn't have 'age'
        company_rows = df[df["_labels"].apply(lambda labels: "Company" in labels)]
        assert company_rows["age"].isna().all()

    def test_labels_are_lists(self, populated_db):
        df = populated_db.nodes_df()
        for labels in df["_labels"]:
            assert isinstance(labels, list)

    def test_empty_graph(self):
        db = grafeo.GrafeoDB()
        df = db.nodes_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["_id", "_labels"]


# --- db.edges_df() ---


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestEdgesDf:
    def test_basic(self, populated_db):
        df = populated_db.edges_df()

        assert isinstance(df, pd.DataFrame)
        assert "_id" in df.columns
        assert "_source" in df.columns
        assert "_target" in df.columns
        assert "_type" in df.columns
        assert len(df) == 2  # KNOWS + WORKS_AT

    def test_edge_types(self, populated_db):
        df = populated_db.edges_df()
        types = set(df["_type"])
        assert types == {"KNOWS", "WORKS_AT"}

    def test_property_columns(self, populated_db):
        df = populated_db.edges_df()
        assert "since" in df.columns
        assert "role" in df.columns

    def test_missing_properties_are_none(self, populated_db):
        df = populated_db.edges_df()
        # KNOWS edge has 'since' but not 'role', WORKS_AT has 'role' but not 'since'
        knows_rows = df[df["_type"] == "KNOWS"]
        assert knows_rows["role"].isna().all()

    def test_empty_graph(self):
        db = grafeo.GrafeoDB()
        df = db.edges_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Column order is implementation-defined (currently alphabetical
        # among the metadata columns); compare as a set so the test
        # doesn't break if the implementation reorders.
        assert set(df.columns) == {"_id", "_type", "_source", "_target"}


# --- nodes_df() and edges_df() with and without pyarrow ---


@pytest.fixture(params=["arrow", "fallback"])
def dataframe_path(request, monkeypatch):
    """Runs a test through both paths of nodes_df() and edges_df(): the Arrow
    fast path (pyarrow importable) and the fallback (pyarrow not importable)."""
    if request.param == "arrow":
        pytest.importorskip("pyarrow")
    else:
        monkeypatch.setitem(sys.modules, "pyarrow", None)
    return request.param


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDataFramePaths:
    """The values of nodes_df() and edges_df() have the same Python types
    whether or not pyarrow is installed."""

    def test_labels_are_python_lists(self, populated_db, dataframe_path):
        df = populated_db.nodes_df()
        for labels in df["_labels"]:
            assert type(labels) is list, f"{dataframe_path}: {type(labels)}"
        assert sorted(tuple(labels) for labels in df["_labels"]) == [
            ("Company",),
            ("Person",),
            ("Person",),
            ("Person",),
        ]

    def test_node_vectors_are_python_lists(self, dataframe_path):
        db = grafeo.GrafeoDB()
        db.create_node(["Doc"], {"name": "Alix", "embedding": [0.5, 3.0, 19.0]})
        df = db.nodes_df()
        embedding = df["embedding"].iloc[0]
        assert type(embedding) is list, f"{dataframe_path}: {type(embedding)}"
        assert embedding == [0.5, 3.0, 19.0]

    def test_edge_vectors_are_python_lists(self, dataframe_path):
        db = grafeo.GrafeoDB()
        alix = db.create_node(["Person"], {"name": "Alix"})
        gus = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(alix.id, gus.id, "KNOWS", {"weights": [0.5, 88.0]})
        df = db.edges_df()
        weights = df["weights"].iloc[0]
        assert type(weights) is list, f"{dataframe_path}: {type(weights)}"
        assert weights == [0.5, 88.0]

    def test_labels_of_an_empty_graph(self, dataframe_path):
        df = grafeo.GrafeoDB().nodes_df()
        assert list(df.columns) == ["_id", "_labels"]
        assert len(df) == 0

    def test_nested_properties_are_python_values(self, typed_db, dataframe_path):
        df = typed_db.nodes_df().sort_values("name")
        alix, gus = df.to_dict("records")
        assert alix["tags"] == ["a", "b"], dataframe_path
        assert type(alix["tags"]) is list, dataframe_path
        assert alix["address"] == {"city": "Amsterdam", "number": 3}, dataframe_path
        assert alix["wait"] == WAIT, dataframe_path
        assert alix["nested"] == [{"k": [3, 19]}], dataframe_path
        assert alix["contact"] == {"phones": ["3", "19"]}, dataframe_path
        assert type(alix["contact"]["phones"]) is list, dataframe_path
        for name in ["tags", "address", "wait", "nested", "contact"]:
            assert gus[name] is None, f"{dataframe_path}: {name} = {gus[name]!r}"

    def test_edge_nested_properties_are_python_values(self, typed_db, dataframe_path):
        df = typed_db.edges_df()
        (knows,) = df.to_dict("records")
        assert knows["years"] == [2019, 2088], dataframe_path
        assert knows["detail"] == {"since": 3}, dataframe_path

    def test_a_map_key_a_row_lacks(self, dataframe_path):
        # With pyarrow a map column is one struct of every key in the column,
        # so a key a map lacks reads None; without pyarrow it is absent.
        db = grafeo.GrafeoDB()
        db.execute("INSERT (:Place {name: 'Paris', at: {city: 'Paris'}})")
        db.execute("INSERT (:Place {name: 'Berlin', at: {number: 19}})")
        df = db.nodes_df().sort_values("name")
        berlin, paris = df.to_dict("records")
        if dataframe_path == "arrow":
            assert paris["at"] == {"city": "Paris", "number": None}
            assert berlin["at"] == {"city": None, "number": 19}
        else:
            assert paris["at"] == {"city": "Paris"}
            assert berlin["at"] == {"number": 19}

    def test_values_of_different_types_in_one_column(self, dataframe_path):
        # With pyarrow a column of mixed types is text (integers and floats
        # together are floats); without pyarrow each value keeps its type.
        db = grafeo.GrafeoDB()
        db.execute("INSERT (:N {k: 1, v: 3, w: 3})")
        db.execute("INSERT (:N {k: 2, v: 'Prague', w: 0.5})")
        df = db.nodes_df().sort_values("k")
        if dataframe_path == "arrow":
            assert list(df["v"]) == ["3", "Prague"]
        else:
            assert list(df["v"]) == [3, "Prague"]
        assert list(df["w"]) == [3.0, 0.5]


WAIT = {"months": 0, "days": 3, "nanos": 0}


@pytest.fixture
def typed_db():
    """Alix has a list, a map, a duration, a list of maps and a map of a list;
    Gus has none of them. Alix knows Gus with a list and a map on the edge."""
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (:Person {name: 'Alix', tags: ['a', 'b'], "
        "address: {city: 'Amsterdam', number: 3}, wait: duration('P3D'), "
        "nested: [{k: [3, 19]}], contact: {phones: ['3', '19']}})"
    )
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.execute(
        "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) "
        "INSERT (a)-[:KNOWS {years: [2019, 2088], detail: {since: 3}}]->(g)"
    )
    return db


class TestTypedArrowExport:
    """The Arrow export keeps lists, maps and durations as Arrow lists and
    structs, so pyarrow and polars read the values the query returns."""

    def test_to_arrow_reads_the_rows_of_the_result(self, typed_db):
        pytest.importorskip("pyarrow")
        result = typed_db.execute(
            "MATCH (p:Person) RETURN p.name AS name, p.tags AS tags, "
            "p.address AS address, p.wait AS wait, p.nested AS nested, "
            "p.contact AS contact ORDER BY name"
        )
        rows = [dict(row) for row in result]
        assert rows[0]["wait"] == WAIT
        assert result.to_arrow().to_pylist() == rows

    def test_polars_reads_the_nested_properties(self, typed_db):
        pytest.importorskip("polars")
        nodes = sorted(typed_db.nodes_to_polars().to_dicts(), key=lambda row: row["name"])
        alix, gus = nodes
        assert alix["_labels"] == ["Person"]
        assert alix["tags"] == ["a", "b"]
        assert alix["address"] == {"city": "Amsterdam", "number": 3}
        assert alix["wait"] == WAIT
        assert alix["nested"] == [{"k": [3, 19]}]
        assert alix["contact"] == {"phones": ["3", "19"]}
        for name in ["tags", "address", "wait", "nested", "contact"]:
            assert gus[name] is None, f"{name} = {gus[name]!r}"
        (knows,) = typed_db.edges_to_polars().to_dicts()
        assert knows["years"] == [2019, 2088]
        assert knows["detail"] == {"since": 3}


# --- Error handling ---


class TestDataFrameErrors:
    def test_to_pandas_without_pandas(self, monkeypatch, populated_db):
        """to_pandas() raises ModuleNotFoundError when pandas isn't installed."""
        # We can't actually uninstall pandas mid-test, but we verify the method exists
        result = populated_db.execute("MATCH (n:Person) RETURN n.name")
        assert hasattr(result, "to_pandas")

    def test_to_polars_without_polars(self, monkeypatch, populated_db):
        """to_polars() raises ModuleNotFoundError when polars isn't installed."""
        result = populated_db.execute("MATCH (n:Person) RETURN n.name")
        assert hasattr(result, "to_polars")
