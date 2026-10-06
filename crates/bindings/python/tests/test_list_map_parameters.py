"""A list, map or bytes parameter returned as it is comes back as given (#574)."""

import grafeo
import pytest

LIST = [1, 2]
MAP = {"city": "Amsterdam", "tags": [1, 2]}

CASES = [
    ("RETURN $x AS v", LIST, [LIST]),
    ("RETURN $x AS v", MAP, [MAP]),
    ("UNWIND [0] AS i WITH $x AS w RETURN w AS v", LIST, [LIST]),
    ("UNWIND [0] AS i WITH $x AS w RETURN w AS v", MAP, [MAP]),
    ("UNWIND $x AS v RETURN v", [MAP, MAP], [MAP, MAP]),
    ("RETURN [$x] AS v", MAP, [[MAP]]),
]


@pytest.mark.parametrize("query, value, expected", CASES)
def test_gql_returns_list_and_map_parameters(query, value, expected):
    db = grafeo.GrafeoDB()
    rows = db.execute(query, {"x": value})
    assert [row["v"] for row in rows] == expected


@pytest.mark.parametrize("query, value, expected", CASES)
def test_cypher_returns_list_and_map_parameters(query, value, expected):
    db = grafeo.GrafeoDB()
    if not hasattr(db, "execute_cypher"):
        pytest.skip("grafeo built without cypher")
    rows = db.execute_cypher(query, {"x": value})
    assert [row["v"] for row in rows] == expected


def test_bytes_parameter_is_returned_like_a_stored_bytes_property():
    # Query rows give bytes as a list of ints, for a stored property too; the
    # parameter must come back the same way, not as ''.
    db = grafeo.GrafeoDB()
    db.create_node(["B"], {"b": b"G\x00\xff"})
    stored = [row["v"] for row in db.execute("MATCH (n:B) RETURN n.b AS v")]
    returned = [row["v"] for row in db.execute("RETURN $x AS v", {"x": b"G\x00\xff"})]
    assert returned == stored
    assert returned != [""]
