"""A value is a node or an edge where the type its cell was built with says so.

A dict with the keys of a node (`{_id: 1, _labels: []}`) is a map, not a node:
`nodes()` used to list it, guessing from the keys. An edge with a property named
`_labels` is an edge. `cell_type(row, column)` says where the nodes and edges
of a row are (experimental, new in 0.6.4).
"""

import pytest
from grafeo import GrafeoDB


@pytest.fixture
def knows():
    db = GrafeoDB()
    db.execute(
        "INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020, _labels: ['L']}]->"
        "(:Person {name: 'Gus'})"
    )
    return db


@pytest.mark.parametrize("language", ["gql", "cypher"])
def test_a_map_with_the_keys_of_a_node_is_a_map(knows, language):
    run = knows.execute if language == "gql" else knows.execute_cypher
    result = run(
        "MATCH (a:Person {name: 'Alix'}) "
        "RETURN a, {_id: 99, _labels: ['Person']} AS fake, "
        "{_id: 98, _type: 'KNOWS', _source: 1, _target: 2} AS fake_edge"
    )
    assert [node["name"] for node in result.nodes()] == ["Alix"]
    assert result.edges() == []
    assert result.cell_type(0, 0) == "node"
    assert result.cell_type(0, 1) != "node"
    assert result.cell_type(0, 2) != "edge"


def test_an_edge_with_a_labels_property_is_an_edge(knows):
    result = knows.execute("MATCH ()-[r:KNOWS]->() RETURN r")
    assert result.nodes() == []
    [edge] = result.edges()
    assert edge.edge_type == "KNOWS"
    assert edge["_labels"] == ["L"]
    assert result.cell_type(0, 0) == "edge"


def test_nested_nodes_are_typed(knows):
    result = knows.execute(
        "MATCH p = (a:Person {name: 'Alix'})-[r]->(b) "
        "RETURN [a, 1] AS l, {k: b} AS m, p, collect(a) AS all"
    )
    assert result.cell_type(0, 0)["tuple"][0] == "node"
    assert result.cell_type(0, 1) == {"struct": {"k": "node"}}
    assert result.cell_type(0, 2) == "path"
    assert result.cell_type(0, 3) == {"list": "node"}
    assert sorted(node["name"] for node in result.nodes()) == ["Alix", "Gus"]
    assert [edge.edge_type for edge in result.edges()] == ["KNOWS"]


def test_each_branch_of_a_union_keeps_its_types(knows):
    result = knows.execute(
        "MATCH (a:Person) RETURN a AS x UNION ALL MATCH ()-[r]->() RETURN r AS x"
    )
    assert [result.cell_type(row, 0) for row in range(len(result))] == [
        "node",
        "node",
        "edge",
    ]


async def test_an_async_result_has_the_cell_types(knows):
    result = await knows.execute_async(
        "MATCH (a:Person {name: 'Alix'}) RETURN a, {_id: 9, _labels: []} AS m"
    )
    assert [node["name"] for node in result.nodes()] == ["Alix"]
    assert result.cell_type(0, 0) == "node"
    assert result.cell_type(0, 1) != "node"
