"""Cypher spec: Functions (openCypher 9 sec 6).

Covers: Aggregation, scalar, string, numeric, list, path, temporal functions.
"""

import math

import grafeo
import pytest

# =============================================================================
# Aggregation Functions (sec 6.1)
# =============================================================================


class TestAggregationFunctions:
    """count, sum, avg, min, max, collect, stDev, percentile."""

    def test_count_star(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        result = list(db.execute_cypher("MATCH (n:N) RETURN count(*) AS r"))
        assert result[0]["r"] == 2

    def test_count_expr(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        result = list(db.execute_cypher("MATCH (n:N) RETURN count(n) AS r"))
        assert result[0]["r"] == 2

    def test_count_distinct(self, db):
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN count(DISTINCT n.v) AS r"))
        assert result[0]["r"] == 2

    def test_sum(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        db.create_node(["N"], {"v": 30})
        result = list(db.execute_cypher("MATCH (n:N) RETURN sum(n.v) AS r"))
        assert result[0]["r"] == 60

    def test_avg(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute_cypher("MATCH (n:N) RETURN avg(n.v) AS r"))
        assert result[0]["r"] == 15.0

    def test_min(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute_cypher("MATCH (n:N) RETURN min(n.v) AS r"))
        assert result[0]["r"] == 5

    def test_max(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute_cypher("MATCH (n:N) RETURN max(n.v) AS r"))
        assert result[0]["r"] == 20

    def test_collect(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 3})
        result = list(db.execute_cypher("MATCH (n:N) RETURN collect(n.v) AS r ORDER BY r"))
        assert sorted(result[0]["r"]) == [1, 2, 3]

    def test_stdev(self, db):
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 4})
        db.create_node(["N"], {"v": 4})
        db.create_node(["N"], {"v": 4})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 7})
        db.create_node(["N"], {"v": 9})
        result = list(db.execute_cypher("MATCH (n:N) RETURN stDev(n.v) AS r"))
        assert abs(result[0]["r"] - 2.0) < 0.5

    def test_stdevp(self, db):
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 4})
        db.create_node(["N"], {"v": 4})
        db.create_node(["N"], {"v": 4})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 7})
        db.create_node(["N"], {"v": 9})
        result = list(db.execute_cypher("MATCH (n:N) RETURN stDevP(n.v) AS r"))
        assert result[0]["r"] > 0

    def test_percentile_cont(self, db):
        for v in [1, 2, 3, 4, 5]:
            db.create_node(["N"], {"v": v})
        result = list(db.execute_cypher("MATCH (n:N) RETURN percentileCont(n.v, 0.5) AS r"))
        assert result[0]["r"] == 3.0

    def test_percentile_disc(self, db):
        for v in [1, 2, 3, 4, 5]:
            db.create_node(["N"], {"v": v})
        result = list(db.execute_cypher("MATCH (n:N) RETURN percentileDisc(n.v, 0.5) AS r"))
        assert result[0]["r"] == 3


# =============================================================================
# Scalar Functions (sec 6.2)
# =============================================================================


class TestScalarFunctions:
    """id, elementId, labels, type, keys, properties, exists, size, length."""

    def test_id(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN id(n) AS r"))
        assert isinstance(result[0]["r"], int)

    def test_element_id(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN elementId(n) AS r"))
        assert result[0]["r"] is not None

    def test_labels(self, db):
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        result = list(db.execute_cypher("MATCH (n:Person) RETURN labels(n) AS r"))
        assert "Person" in result[0]["r"]
        assert "Developer" in result[0]["r"]

    def test_type(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute_cypher("MATCH ()-[r:KNOWS]->() RETURN type(r) AS r"))
        assert result[0]["r"] == "KNOWS"

    def test_keys(self, db):
        db.create_node(["N"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("MATCH (n:N) RETURN keys(n) AS r"))
        assert "name" in result[0]["r"]
        assert "age" in result[0]["r"]

    def test_properties(self, db):
        db.create_node(["N"], {"name": "Alix", "age": 30})
        result = list(db.execute_cypher("MATCH (n:N) RETURN properties(n) AS r"))
        assert result[0]["r"]["name"] == "Alix"
        assert result[0]["r"]["age"] == 30

    def test_exists_property(self, db):
        """exists(n.prop) function form."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"w": 2})
        result = list(db.execute_cypher("MATCH (n:N) WHERE exists(n.v) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_size_list(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN size([1, 2, 3]) AS r"))
        assert result[0]["r"] == 3

    def test_size_string(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN size(n.v) AS r"))
        assert result[0]["r"] == 5

    def test_length_string(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN length(n.v) AS r"))
        assert result[0]["r"] == 5

    def test_to_integer(self, db):
        db.create_node(["N"], {"v": "42"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN toInteger(n.v) AS r"))
        assert result[0]["r"] == 42

    def test_to_float(self, db):
        db.create_node(["N"], {"v": "3.14"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN toFloat(n.v) AS r"))
        assert abs(result[0]["r"] - 3.14) < 0.001

    def test_to_string(self, db):
        db.create_node(["N"], {"v": 42})
        result = list(db.execute_cypher("MATCH (n:N) RETURN toString(n.v) AS r"))
        assert result[0]["r"] == "42"

    def test_to_boolean(self, db):
        db.create_node(["N"], {"v": "true"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN toBoolean(n.v) AS r"))
        assert result[0]["r"] is True


# =============================================================================
# String Functions (sec 6.3)
# =============================================================================


class TestStringFunctions:
    """toUpper, toLower, trim, replace, substring, split, left, right, reverse."""

    def test_to_upper(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN toUpper(n.v) AS r"))
        assert result[0]["r"] == "HELLO"

    def test_to_lower(self, db):
        db.create_node(["N"], {"v": "HELLO"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN toLower(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_trim(self, db):
        db.create_node(["N"], {"v": "  hello  "})
        result = list(db.execute_cypher("MATCH (n:N) RETURN trim(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_ltrim(self, db):
        db.create_node(["N"], {"v": "  hello"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN ltrim(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_rtrim(self, db):
        db.create_node(["N"], {"v": "hello  "})
        result = list(db.execute_cypher("MATCH (n:N) RETURN rtrim(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_replace(self, db):
        db.create_node(["N"], {"v": "hello world"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN replace(n.v, 'world', 'grafeo') AS r"))
        assert result[0]["r"] == "hello grafeo"

    def test_substring(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN substring(n.v, 0, 5) AS r"))
        assert result[0]["r"] == "Amste"

    def test_split(self, db):
        db.create_node(["N"], {"v": "a,b,c"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN split(n.v, ',') AS r"))
        assert result[0]["r"] == ["a", "b", "c"]

    def test_left(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN left(n.v, 3) AS r"))
        assert result[0]["r"] == "Ams"

    def test_right(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN right(n.v, 3) AS r"))
        assert result[0]["r"] == "dam"

    def test_reverse(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN reverse(n.v) AS r"))
        assert result[0]["r"] == "olleh"

    def test_char_length(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN char_length(n.v) AS r"))
        assert result[0]["r"] == 5


# =============================================================================
# Numeric Functions (sec 6.4)
# =============================================================================


class TestNumericFunctions:
    """abs, ceil, floor, round, sqrt, sign, log, exp, trig."""

    def test_abs(self, db):
        db.create_node(["N"], {"v": -5})
        result = list(db.execute_cypher("MATCH (n:N) RETURN abs(n.v) AS r"))
        assert result[0]["r"] == 5

    def test_ceil(self, db):
        db.create_node(["N"], {"v": 2.3})
        result = list(db.execute_cypher("MATCH (n:N) RETURN ceil(n.v) AS r"))
        assert result[0]["r"] == 3

    def test_floor(self, db):
        db.create_node(["N"], {"v": 2.7})
        result = list(db.execute_cypher("MATCH (n:N) RETURN floor(n.v) AS r"))
        assert result[0]["r"] == 2

    def test_round(self, db):
        db.create_node(["N"], {"v": 2.5})
        result = list(db.execute_cypher("MATCH (n:N) RETURN round(n.v) AS r"))
        assert result[0]["r"] == 3

    def test_sqrt(self, db):
        db.create_node(["N"], {"v": 9})
        result = list(db.execute_cypher("MATCH (n:N) RETURN sqrt(n.v) AS r"))
        assert result[0]["r"] == 3.0

    def test_sign(self, db):
        db.create_node(["N"], {"v": -5})
        result = list(db.execute_cypher("MATCH (n:N) RETURN sign(n.v) AS r"))
        assert result[0]["r"] == -1

    def test_log(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN log(e()) AS r"))
        assert abs(result[0]["r"] - 1.0) < 0.001

    def test_log10(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN log10(100) AS r"))
        assert abs(result[0]["r"] - 2.0) < 0.001

    def test_exp(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN exp(1) AS r"))
        assert abs(result[0]["r"] - math.e) < 0.001

    def test_e(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN e() AS r"))
        assert abs(result[0]["r"] - math.e) < 0.001

    def test_pi(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN pi() AS r"))
        assert abs(result[0]["r"] - math.pi) < 0.001

    def test_rand(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN rand() AS r"))
        assert 0.0 <= result[0]["r"] <= 1.0

    def test_sin(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN sin(0) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_cos(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN cos(0) AS r"))
        assert abs(result[0]["r"] - 1.0) < 0.001

    def test_tan(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN tan(0) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_asin(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN asin(1) AS r"))
        assert abs(result[0]["r"] - math.pi / 2) < 0.001

    def test_acos(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN acos(1) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_atan(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN atan(1) AS r"))
        assert abs(result[0]["r"] - math.pi / 4) < 0.001

    def test_atan2(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN atan2(1, 1) AS r"))
        assert abs(result[0]["r"] - math.pi / 4) < 0.001

    def test_degrees(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN degrees(pi()) AS r"))
        assert abs(result[0]["r"] - 180.0) < 0.001

    def test_radians(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN radians(180) AS r"))
        assert abs(result[0]["r"] - math.pi) < 0.001


# =============================================================================
# List Functions (sec 6.5)
# =============================================================================


class TestListFunctions:
    """head, last, tail, size, range, reverse, reduce, keys."""

    def test_head(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN head([10, 20, 30]) AS r"))
        assert result[0]["r"] == 10

    def test_last(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN last([10, 20, 30]) AS r"))
        assert result[0]["r"] == 30

    def test_tail(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN tail([10, 20, 30]) AS r"))
        assert result[0]["r"] == [20, 30]

    def test_size(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN size([10, 20, 30]) AS r"))
        assert result[0]["r"] == 3

    def test_range(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN range(1, 5) AS r"))
        assert result[0]["r"] == [1, 2, 3, 4, 5]

    def test_range_with_step(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN range(0, 10, 3) AS r"))
        assert result[0]["r"] == [0, 3, 6, 9]

    def test_reverse_list(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN reverse([1, 2, 3]) AS r"))
        assert result[0]["r"] == [3, 2, 1]

    def test_keys_map(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN keys({a: 1, b: 2}) AS r"))
        assert sorted(result[0]["r"]) == ["a", "b"]


# =============================================================================
# Path Functions (sec 6.6)
# =============================================================================


class TestPathFunctions:
    """length, nodes, relationships on paths."""

    def test_length_path(self, db):
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        c = db.create_node(["N"], {"v": 3})
        db.create_edge(a.id, b.id, "NEXT")
        db.create_edge(b.id, c.id, "NEXT")
        result = list(
            db.execute_cypher(
                "MATCH p = (a:N {v: 1})-[:NEXT*2]->(b:N {v: 3}) RETURN length(p) AS r"
            )
        )
        assert result[0]["r"] == 2

    def test_nodes_path(self, db):
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute_cypher("MATCH p = (a:N {v: 1})-[:NEXT]->(b) RETURN size(nodes(p)) AS r")
        )
        assert result[0]["r"] == 2

    def test_relationships_path(self, db):
        a = db.create_node(["N"], {"v": 1})
        b = db.create_node(["N"], {"v": 2})
        db.create_edge(a.id, b.id, "NEXT")
        result = list(
            db.execute_cypher(
                "MATCH p = (a:N {v: 1})-[:NEXT]->(b) RETURN size(relationships(p)) AS r"
            )
        )
        assert result[0]["r"] == 1


# =============================================================================
# Temporal Functions (sec 6.7)
# =============================================================================


class TestTemporalFunctions:
    """date, time, datetime, duration, now, temporal accessors."""

    def test_date_current(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN date() AS r"))
        assert result[0]["r"] is not None

    def test_date_parse(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN date('2024-01-15') AS r"))
        assert "2024-01-15" in str(result[0]["r"])

    def test_time_parse(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN time('14:30:00') AS r"))
        assert "14:30" in str(result[0]["r"])

    def test_datetime_parse(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN datetime('2024-01-15T14:30:00') AS r"))
        assert "2024" in str(result[0]["r"])

    def test_duration_parse(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN duration('P1Y2M') AS r"))
        assert result[0]["r"] is not None

    def test_now(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN now() AS r"))
        assert result[0]["r"] is not None

    def test_year_accessor(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN year(date('2024-03-15')) AS r"))
        assert result[0]["r"] == 2024

    def test_month_accessor(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN month(date('2024-03-15')) AS r"))
        assert result[0]["r"] == 3

    def test_day_accessor(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN day(date('2024-03-15')) AS r"))
        assert result[0]["r"] == 15

    @pytest.mark.xfail(
        strict=True,
        raises=grafeo.GrafeoError,
        reason="spec: #534 (point values and distance); issue: #534",
    )
    def test_point_2d(self, db):
        """point({x, y}) 2D Cartesian."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN point({x: 1.0, y: 2.0}) AS r"))
        assert result[0]["r"] is not None

    @pytest.mark.xfail(
        strict=True,
        raises=grafeo.GrafeoError,
        reason="spec: #534 (point values and distance); issue: #534",
    )
    def test_distance(self, db):
        """distance(point1, point2)."""
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute_cypher(
                "MATCH (n:N) RETURN distance(point({x: 0, y: 0}), point({x: 3, y: 4})) AS r"
            )
        )
        assert result[0]["r"] == 5.0
