"""GQL spec: Functions (ISO sec 20).

Covers: aggregation, string, numeric, temporal, list, graph element, path,
type conversion functions.
"""

import math

# =============================================================================
# Aggregation Functions
# =============================================================================


class TestAggregationFunctions:
    """COUNT, SUM, AVG, MIN, MAX, COLLECT, STDEV, PERCENTILE, LISTAGG."""

    def test_count_star(self, db):
        """COUNT(n) counts all rows."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        result = list(db.execute("MATCH (n:N) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_count_expr(self, db):
        """COUNT(expr) counts non-null values."""
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {})
        result = list(db.execute("MATCH (n:N) RETURN count(n.v) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_count_distinct(self, db):
        """COUNT(DISTINCT expr) counts unique values."""
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        result = list(db.execute("MATCH (n:N) RETURN count(DISTINCT n.v) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_sum(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        db.create_node(["N"], {"v": 30})
        result = list(db.execute("MATCH (n:N) RETURN sum(n.v) AS s"))
        assert result[0]["s"] == 60

    def test_avg(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute("MATCH (n:N) RETURN avg(n.v) AS a"))
        assert abs(result[0]["a"] - 15.0) < 0.001

    def test_min(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute("MATCH (n:N) RETURN min(n.v) AS m"))
        assert result[0]["m"] == 5

    def test_max(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute("MATCH (n:N) RETURN max(n.v) AS m"))
        assert result[0]["m"] == 20

    def test_collect(self, db):
        """COLLECT aggregates into a list."""
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        result = list(db.execute("MATCH (n:N) RETURN collect(n.v) AS items"))
        assert set(result[0]["items"]) == {"a", "b"}

    def test_stdev(self, db):
        """STDEV() sample standard deviation."""
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            db.create_node(["N"], {"v": v})
        result = list(db.execute("MATCH (n:N) RETURN stdev(n.v) AS s"))
        assert abs(result[0]["s"] - 2.0) < 0.2

    def test_stdevp(self, db):
        """STDEVP() population standard deviation."""
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            db.create_node(["N"], {"v": v})
        result = list(db.execute("MATCH (n:N) RETURN stdevp(n.v) AS s"))
        assert result[0]["s"] > 0

    def test_percentile_disc(self, db):
        """PERCENTILE_DISC discrete percentile."""
        for v in [1, 2, 3, 4, 5]:
            db.create_node(["N"], {"v": v})
        result = list(db.execute("MATCH (n:N) RETURN percentile_disc(n.v, 0.5) AS p"))
        assert result[0]["p"] == 3

    def test_percentile_cont(self, db):
        """PERCENTILE_CONT continuous percentile."""
        for v in [1, 2, 3, 4, 5]:
            db.create_node(["N"], {"v": v})
        result = list(db.execute("MATCH (n:N) RETURN percentile_cont(n.v, 0.5) AS p"))
        assert abs(result[0]["p"] - 3.0) < 0.1

    def test_listagg(self, db):
        """LISTAGG string aggregation."""
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        db.create_node(["N"], {"v": "c"})
        result = list(db.execute("MATCH (n:N) RETURN listagg(n.v, ', ') AS r"))
        r = result[0]["r"]
        # Order may vary, but should contain all values
        assert "a" in r
        assert "b" in r
        assert "c" in r


# =============================================================================
# String Functions
# =============================================================================


class TestStringFunctions:
    """String manipulation functions."""

    def test_to_upper(self, db):
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute("MATCH (n:N) RETURN toUpper(n.v) AS r"))
        assert result[0]["r"] == "HELLO"

    def test_upper_alias(self, db):
        """upper() alias for toUpper()."""
        db.create_node(["N"], {"v": "hello"})
        result = list(db.execute("MATCH (n:N) RETURN upper(n.v) AS r"))
        assert result[0]["r"] == "HELLO"

    def test_to_lower(self, db):
        db.create_node(["N"], {"v": "HELLO"})
        result = list(db.execute("MATCH (n:N) RETURN toLower(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_lower_alias(self, db):
        db.create_node(["N"], {"v": "HELLO"})
        result = list(db.execute("MATCH (n:N) RETURN lower(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_trim(self, db):
        db.create_node(["N"], {"v": "  hello  "})
        result = list(db.execute("MATCH (n:N) RETURN trim(n.v) AS r"))
        assert result[0]["r"] == "hello"

    def test_ltrim(self, db):
        db.create_node(["N"], {"v": "  hello  "})
        result = list(db.execute("MATCH (n:N) RETURN ltrim(n.v) AS r"))
        assert result[0]["r"] == "hello  "

    def test_rtrim(self, db):
        db.create_node(["N"], {"v": "  hello  "})
        result = list(db.execute("MATCH (n:N) RETURN rtrim(n.v) AS r"))
        assert result[0]["r"] == "  hello"

    def test_replace(self, db):
        db.create_node(["N"], {"v": "hello world"})
        result = list(db.execute("MATCH (n:N) RETURN replace(n.v, 'world', 'grafeo') AS r"))
        assert result[0]["r"] == "hello grafeo"

    def test_substring(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute("MATCH (n:N) RETURN substring(n.v, 0, 5) AS r"))
        assert result[0]["r"] == "Amste"

    def test_split(self, db):
        db.create_node(["N"], {"v": "a,b,c"})
        result = list(db.execute("MATCH (n:N) RETURN split(n.v, ',') AS r"))
        assert result[0]["r"] == ["a", "b", "c"]

    def test_left(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute("MATCH (n:N) RETURN left(n.v, 3) AS r"))
        assert result[0]["r"] == "Ams"

    def test_right(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute("MATCH (n:N) RETURN right(n.v, 3) AS r"))
        assert result[0]["r"] == "dam"

    def test_char_length(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(db.execute("MATCH (n:N) RETURN char_length(n.v) AS r"))
        assert result[0]["r"] == 9

    def test_reverse_string(self, db):
        db.create_node(["N"], {"v": "abc"})
        result = list(db.execute("MATCH (n:N) RETURN reverse(n.v) AS r"))
        assert result[0]["r"] == "cba"

    def test_to_string(self, db):
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) RETURN toString(n.v) AS r"))
        assert result[0]["r"] == "42"

    def test_string_join(self, db):
        """string_join(list, sep) list-to-string concatenation."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN string_join(['a', 'b', 'c'], '-') AS r"))
        assert result[0]["r"] == "a-b-c"

    def test_octet_length(self, db):
        """octet_length(str) byte length."""
        db.create_node(["N"], {"v": "abc"})
        result = list(db.execute("MATCH (n:N) RETURN octet_length(n.v) AS r"))
        assert result[0]["r"] == 3


# =============================================================================
# Numeric Functions
# =============================================================================


class TestNumericFunctions:
    """Numeric math functions."""

    def test_abs(self, db):
        db.create_node(["N"], {"v": -5})
        result = list(db.execute("MATCH (n:N) RETURN abs(n.v) AS r"))
        assert result[0]["r"] == 5

    def test_ceil(self, db):
        db.create_node(["N"], {"v": 3.2})
        result = list(db.execute("MATCH (n:N) RETURN ceil(n.v) AS r"))
        assert result[0]["r"] == 4

    def test_ceiling_alias(self, db):
        db.create_node(["N"], {"v": 3.2})
        result = list(db.execute("MATCH (n:N) RETURN ceiling(n.v) AS r"))
        assert result[0]["r"] == 4

    def test_floor(self, db):
        db.create_node(["N"], {"v": 3.8})
        result = list(db.execute("MATCH (n:N) RETURN floor(n.v) AS r"))
        assert result[0]["r"] == 3

    def test_round(self, db):
        db.create_node(["N"], {"v": 3.5})
        result = list(db.execute("MATCH (n:N) RETURN round(n.v) AS r"))
        assert result[0]["r"] == 4

    def test_sqrt(self, db):
        db.create_node(["N"], {"v": 16})
        result = list(db.execute("MATCH (n:N) RETURN sqrt(n.v) AS r"))
        assert result[0]["r"] == 4.0

    def test_sign(self, db):
        db.create_node(["N"], {"v": -5})
        result = list(db.execute("MATCH (n:N) RETURN sign(n.v) AS r"))
        assert result[0]["r"] == -1

    def test_log_ln(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN log(e()) AS r"))
        assert abs(result[0]["r"] - 1.0) < 0.001

    def test_log10(self, db):
        db.create_node(["N"], {"v": 100})
        result = list(db.execute("MATCH (n:N) RETURN log10(n.v) AS r"))
        assert abs(result[0]["r"] - 2.0) < 0.001

    def test_exp(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN exp(n.v) AS r"))
        assert abs(result[0]["r"] - math.e) < 0.001

    def test_rand(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN rand() AS r"))
        assert 0 <= result[0]["r"] <= 1

    def test_sin(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN sin(0) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_cos(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN cos(0) AS r"))
        assert abs(result[0]["r"] - 1.0) < 0.001

    def test_tan(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN tan(0) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_asin(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN asin(1) AS r"))
        assert abs(result[0]["r"] - math.pi / 2) < 0.001

    def test_acos(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN acos(1) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_atan(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN atan(0) AS r"))
        assert abs(result[0]["r"]) < 0.001

    def test_degrees(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN degrees(pi()) AS r"))
        assert abs(result[0]["r"] - 180.0) < 0.001

    def test_radians(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN radians(180) AS r"))
        assert abs(result[0]["r"] - math.pi) < 0.001

    def test_pi(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN pi() AS r"))
        assert abs(result[0]["r"] - math.pi) < 0.001

    def test_e(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN e() AS r"))
        assert abs(result[0]["r"] - math.e) < 0.001


# =============================================================================
# Temporal Functions
# =============================================================================


class TestTemporalFunctions:
    """Temporal construction and extraction."""

    def test_date_no_arg(self, db):
        """date() returns today."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN date() AS r"))
        assert result[0]["r"] is not None

    def test_date_parse(self, db):
        """date(string) parses ISO date."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN date('2024-06-15') AS r"))
        assert "2024-06-15" in str(result[0]["r"])

    def test_time_no_arg(self, db):
        """time() returns current time."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN time() AS r"))
        assert result[0]["r"] is not None

    def test_time_parse(self, db):
        """time(string) parses ISO time."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN time('14:30:00') AS r"))
        assert "14:30" in str(result[0]["r"])

    def test_datetime_parse(self, db):
        """datetime(string) parses ISO datetime."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN datetime('2024-06-15T14:30:00') AS r"))
        assert "2024" in str(result[0]["r"])

    def test_duration_parse(self, db):
        """duration(string) parses ISO duration."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN duration('P1Y2M3D') AS r"))
        assert result[0]["r"] is not None

    def test_current_date(self, db):
        """current_date() alias."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN current_date() AS r"))
        assert result[0]["r"] is not None

    def test_current_time(self, db):
        """current_time() alias."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN current_time() AS r"))
        assert result[0]["r"] is not None

    def test_now(self, db):
        """now() returns current timestamp."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN now() AS r"))
        assert result[0]["r"] is not None

    def test_timestamp_returns_millis(self, db):
        """timestamp() returns millisecond epoch as integer (#179)."""
        result = list(db.execute("RETURN timestamp() AS ts"))
        assert len(result) == 1
        assert result[0]["ts"] is not None
        assert isinstance(result[0]["ts"], int)
        assert result[0]["ts"] > 1_577_836_800_000  # after 2020-01-01

    def test_timestamp_in_on_create_set(self, db):
        """MERGE ON CREATE SET n.ts = timestamp() stores non-null value (#179)."""
        db.execute("MERGE (n:T {id: 'x'}) ON CREATE SET n.ts = timestamp()")
        result = list(db.execute("MATCH (n:T) RETURN n.ts"))
        assert result[0]["n.ts"] is not None
        assert result[0]["n.ts"] > 1_577_836_800_000

    def test_year_extraction(self, db):
        """year(date) extracts year."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN year(date('2024-06-15')) AS r"))
        assert result[0]["r"] == 2024

    def test_month_extraction(self, db):
        """month(date) extracts month."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN month(date('2024-06-15')) AS r"))
        assert result[0]["r"] == 6

    def test_day_extraction(self, db):
        """day(date) extracts day."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN day(date('2024-06-15')) AS r"))
        assert result[0]["r"] == 15

    def test_hour_extraction(self, db):
        """hour(time) extracts hour."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN hour(time('14:30:00')) AS r"))
        assert result[0]["r"] == 14

    def test_minute_extraction(self, db):
        """minute(time) extracts minute."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN minute(time('14:30:00')) AS r"))
        assert result[0]["r"] == 30

    def test_second_extraction(self, db):
        """second(time) extracts second."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN second(time('14:30:45')) AS r"))
        assert result[0]["r"] == 45

    def test_temporal_arithmetic(self, db):
        """date + duration temporal arithmetic."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN date('2024-01-15') + duration('P10D') AS r"))
        assert "2024-01-25" in str(result[0]["r"])

    def test_local_time_alias(self, db):
        """local_time() is alias for time()."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN local_time() AS r"))
        assert result[0]["r"] is not None

    def test_local_datetime_alias(self, db):
        """local_datetime() is alias for datetime()."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN local_datetime() AS r"))
        assert result[0]["r"] is not None

    def test_zoned_datetime_no_arg(self, db):
        """zoned_datetime() returns current UTC."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN zoned_datetime() AS r"))
        assert result[0]["r"] is not None

    def test_date_trunc(self, db):
        """date_trunc(unit, temporal) truncation."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN date_trunc('month', date('2024-06-15')) AS r"))
        assert "2024-06-01" in str(result[0]["r"])


# =============================================================================
# List Functions
# =============================================================================


class TestListFunctions:
    """List manipulation functions."""

    def test_head(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN head([1, 2, 3]) AS r"))
        assert result[0]["r"] == 1

    def test_tail(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN tail([1, 2, 3]) AS r"))
        assert result[0]["r"] == [2, 3]

    def test_last(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN last([1, 2, 3]) AS r"))
        assert result[0]["r"] == 3

    def test_reverse_list(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN reverse([1, 2, 3]) AS r"))
        assert result[0]["r"] == [3, 2, 1]

    def test_size(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN size([1, 2, 3]) AS r"))
        assert result[0]["r"] == 3

    def test_range_two_args(self, db):
        """range(start, end)."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN range(1, 5) AS r"))
        assert result[0]["r"] == [1, 2, 3, 4, 5]

    def test_range_with_step(self, db):
        """range(start, end, step)."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN range(0, 10, 3) AS r"))
        assert result[0]["r"] == [0, 3, 6, 9]

    def test_all_predicate(self, db):
        """all(x IN list WHERE pred) list predicate."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN all(x IN [2, 4, 6] WHERE x % 2 = 0) AS r"))
        assert result[0]["r"] is True

    def test_any_predicate(self, db):
        """any(x IN list WHERE pred) list predicate."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN any(x IN [1, 2, 3] WHERE x > 2) AS r"))
        assert result[0]["r"] is True

    def test_none_predicate(self, db):
        """none(x IN list WHERE pred) list predicate."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN none(x IN [1, 2, 3] WHERE x > 10) AS r"))
        assert result[0]["r"] is True

    def test_single_predicate(self, db):
        """single(x IN list WHERE pred) exactly one match."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN single(x IN [1, 2, 3] WHERE x = 2) AS r"))
        assert result[0]["r"] is True


# =============================================================================
# Graph Element Functions
# =============================================================================


class TestGraphElementFunctions:
    """id, element_id, labels, type, keys, properties, hasLabel."""

    def test_id(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) RETURN id(n) AS r"))
        assert isinstance(result[0]["r"], int)

    def test_element_id(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) RETURN element_id(n) AS r"))
        assert result[0]["r"] is not None

    def test_labels(self, db):
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) RETURN labels(n) AS r"))
        assert "Person" in result[0]["r"]
        assert "Developer" in result[0]["r"]

    def test_type_edge(self, db):
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(db.execute("MATCH ()-[e:KNOWS]->() RETURN type(e) AS r"))
        assert result[0]["r"] == "KNOWS"

    def test_keys(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:Person) RETURN keys(n) AS r"))
        assert "name" in result[0]["r"]
        assert "age" in result[0]["r"]

    def test_properties(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(db.execute("MATCH (n:Person) RETURN properties(n) AS r"))
        assert result[0]["r"]["name"] == "Alix"
        assert result[0]["r"]["age"] == 30

    def test_has_label(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:Person) RETURN hasLabel(n, 'Person') AS r"))
        assert result[0]["r"] is True

    def test_start_node(self, db):
        """startNode(r) returns the source node (0.6.0; its ID before)."""
        db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        result = list(db.execute("MATCH ()-[r:KNOWS]->() RETURN startNode(r) AS sn"))
        assert len(result) == 1
        assert result[0]["sn"]["name"] == "Alix"
        assert result[0]["sn"]["_labels"] == ["Person"]

    def test_end_node(self, db):
        """endNode(r) returns the destination node (0.6.0; its ID before)."""
        db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        result = list(db.execute("MATCH ()-[r:KNOWS]->() RETURN endNode(r) AS en"))
        assert len(result) == 1
        assert result[0]["en"]["name"] == "Gus"
        assert result[0]["en"]["_labels"] == ["Person"]

    def test_start_node_matches_source(self, db):
        """startNode(r) == id(s) and endNode(r) == id(t) (#180)."""
        db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        result = list(
            db.execute(
                "MATCH (s)-[r:KNOWS]->(t) "
                "RETURN startNode(r) = id(s) AS src_ok, endNode(r) = id(t) AS dst_ok"
            )
        )
        assert result[0]["src_ok"] is True
        assert result[0]["dst_ok"] is True


# =============================================================================
# Type Conversion Functions
# =============================================================================


class TestTypeConversion:
    """toInteger, toFloat, toString, toBoolean, toDate, toTime, etc."""

    def test_to_integer(self, db):
        db.create_node(["N"], {"v": "42"})
        result = list(db.execute("MATCH (n:N) RETURN toInteger(n.v) AS r"))
        assert result[0]["r"] == 42

    def test_to_int_alias(self, db):
        db.create_node(["N"], {"v": "42"})
        result = list(db.execute("MATCH (n:N) RETURN toInt(n.v) AS r"))
        assert result[0]["r"] == 42

    def test_to_float(self, db):
        db.create_node(["N"], {"v": "3.14"})
        result = list(db.execute("MATCH (n:N) RETURN toFloat(n.v) AS r"))
        assert abs(result[0]["r"] - 3.14) < 0.001

    def test_to_boolean(self, db):
        db.create_node(["N"], {"v": "true"})
        result = list(db.execute("MATCH (n:N) RETURN toBoolean(n.v) AS r"))
        assert result[0]["r"] is True

    def test_to_date(self, db):
        db.create_node(["N"], {"v": "2024-06-15"})
        result = list(db.execute("MATCH (n:N) RETURN toDate(n.v) AS r"))
        assert "2024-06-15" in str(result[0]["r"])

    def test_to_time(self, db):
        db.create_node(["N"], {"v": "14:30:00"})
        result = list(db.execute("MATCH (n:N) RETURN toTime(n.v) AS r"))
        assert "14:30" in str(result[0]["r"])

    def test_to_datetime(self, db):
        db.create_node(["N"], {"v": "2024-06-15T14:30:00"})
        result = list(db.execute("MATCH (n:N) RETURN toDatetime(n.v) AS r"))
        assert "2024" in str(result[0]["r"])

    def test_to_duration(self, db):
        db.create_node(["N"], {"v": "P1Y2M"})
        result = list(db.execute("MATCH (n:N) RETURN toDuration(n.v) AS r"))
        assert result[0]["r"] is not None

    def test_to_list(self, db):
        """toList wraps scalar in list."""
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) RETURN toList(n.v) AS r"))
        assert result[0]["r"] == [42]
