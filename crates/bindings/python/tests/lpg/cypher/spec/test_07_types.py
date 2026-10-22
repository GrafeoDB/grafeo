"""Cypher spec: Types (openCypher 9 sec 7).

Covers: Integer, Float, String, Boolean, Null, List, Map, Date, Time,
DateTime, Duration, Path, Node, Relationship. Spatial types (Point) are
marked xfail as they are not implemented.
"""

import pytest

# =============================================================================
# Scalar Types
# =============================================================================


class TestScalarTypes:
    """Integer, Float, String, Boolean, Null literals and type behavior."""

    def test_integer_decimal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN 42 AS r"))
        assert result[0]["r"] == 42

    def test_integer_hex(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN 0xFF AS r"))
        assert result[0]["r"] == 255

    def test_integer_octal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN 0o77 AS r"))
        assert result[0]["r"] == 63

    def test_float(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN 3.14 AS r"))
        assert abs(result[0]["r"] - 3.14) < 0.001

    def test_float_scientific(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN 1.5e2 AS r"))
        assert result[0]["r"] == 150.0

    def test_string_single_quoted(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN 'hello' AS r"))
        assert result[0]["r"] == "hello"

    def test_string_double_quoted(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher('MATCH (n:N) RETURN "hello" AS r'))
        assert result[0]["r"] == "hello"

    def test_boolean_true(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN true AS r"))
        assert result[0]["r"] is True

    def test_boolean_false(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN false AS r"))
        assert result[0]["r"] is False

    def test_null(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN null AS r"))
        assert result[0]["r"] is None


# =============================================================================
# Collection Types
# =============================================================================


class TestCollectionTypes:
    """List and Map types."""

    def test_list_heterogeneous(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN [1, 'two', 3.0, true] AS r"))
        r = result[0]["r"]
        assert r[0] == 1
        assert r[1] == "two"
        assert r[3] is True

    def test_map_type(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN {name: 'Alix', age: 30} AS r"))
        assert result[0]["r"]["name"] == "Alix"
        assert result[0]["r"]["age"] == 30


# =============================================================================
# Temporal Types
# =============================================================================


class TestTemporalTypes:
    """Date, Time, DateTime, Duration types."""

    def test_date_type(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN date('2024-01-15') AS r"))
        assert "2024-01-15" in str(result[0]["r"])

    def test_time_type(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN time('14:30:00') AS r"))
        assert "14:30" in str(result[0]["r"])

    def test_datetime_type(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN datetime('2024-01-15T14:30:00') AS r"))
        assert "2024" in str(result[0]["r"])

    def test_duration_type(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN duration('P1Y2M3D') AS r"))
        assert result[0]["r"] is not None


# =============================================================================
# Graph Element Types
# =============================================================================


class TestGraphElementTypes:
    """Node, Relationship, Path types."""

    def test_node_return(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n"))
        assert len(result) == 1
        # Node returned as dict-like with properties
        assert result[0]["n"] is not None

    def test_relationship_return(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS", {"since": 2020})
        result = list(db.execute_cypher("MATCH ()-[r:KNOWS]->() RETURN r"))
        assert len(result) == 1

    def test_path_return(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute_cypher("MATCH p = (:Person)-[:KNOWS]->(:Person) RETURN p"))
        assert len(result) == 1


# =============================================================================
# Spatial Types (MISSING)
# =============================================================================


class TestSpatialTypes:
    """Point types: not implemented."""

    @pytest.mark.xfail(reason="Spatial Point type not implemented")
    def test_point_2d_cartesian(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN point({x: 1.0, y: 2.0}) AS r"))
        assert result[0]["r"] is not None

    @pytest.mark.xfail(reason="Spatial Point type not implemented")
    def test_point_wgs84(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute_cypher("MATCH (n:N) RETURN point({latitude: 52.37, longitude: 4.89}) AS r")
        )
        assert result[0]["r"] is not None

    @pytest.mark.xfail(reason="Spatial Point type not implemented")
    def test_point_3d(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN point({x: 1.0, y: 2.0, z: 3.0}) AS r"))
        assert result[0]["r"] is not None
