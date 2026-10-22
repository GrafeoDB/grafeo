"""GQL spec: Expressions (ISO sec 20).

Covers: arithmetic, comparison, logical operators, string predicates,
CASE, NULLIF, COALESCE, CAST, LET IN END, list comprehensions, reduce,
literals, parameters, property access, SESSION_USER.
"""


# =============================================================================
# Operators (sec 20)
# =============================================================================


class TestArithmeticOperators:
    """Arithmetic: +, -, *, /, %."""

    def test_addition(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN n.v + 5 AS r"))
        assert result[0]["r"] == 15

    def test_subtraction(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN n.v - 3 AS r"))
        assert result[0]["r"] == 7

    def test_multiplication(self, db):
        db.create_node(["N"], {"v": 6})
        result = list(db.execute("MATCH (n:N) RETURN n.v * 7 AS r"))
        assert result[0]["r"] == 42

    def test_division(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN n.v / 2 AS r"))
        assert result[0]["r"] == 5

    def test_modulo(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN n.v % 3 AS r"))
        assert result[0]["r"] == 1

    def test_unary_minus(self, db):
        db.create_node(["N"], {"v": 5})
        result = list(db.execute("MATCH (n:N) RETURN -n.v AS r"))
        assert result[0]["r"] == -5

    def test_unary_plus(self, db):
        db.create_node(["N"], {"v": 5})
        result = list(db.execute("MATCH (n:N) RETURN +n.v AS r"))
        assert result[0]["r"] == 5

    def test_string_concatenation_pipe(self, db):
        """|| ISO string concatenation operator."""
        db.create_node(["N"], {"a": "hello", "b": "world"})
        result = list(db.execute("MATCH (n:N) RETURN n.a || ' ' || n.b AS r"))
        assert result[0]["r"] == "hello world"

    def test_string_concatenation(self, db):
        """+ string concatenation (works as alternative to ||)."""
        db.create_node(["N"], {"a": "hello", "b": "world"})
        result = list(db.execute("MATCH (n:N) RETURN n.a + ' ' + n.b AS r"))
        assert result[0]["r"] == "hello world"


class TestComparisonOperators:
    """Comparison: =, <>, <, <=, >, >=."""

    def test_equals(self, db):
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) WHERE n.v = 10 RETURN n.v"))
        assert len(result) == 1

    def test_not_equals(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute("MATCH (n:N) WHERE n.v <> 10 RETURN n.v"))
        assert len(result) == 1
        assert result[0]["n.v"] == 20

    def test_less_than(self, db):
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 15})
        result = list(db.execute("MATCH (n:N) WHERE n.v < 10 RETURN n.v"))
        assert result[0]["n.v"] == 5

    def test_less_equal(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 20})
        result = list(db.execute("MATCH (n:N) WHERE n.v <= 10 RETURN n.v"))
        assert len(result) == 1

    def test_greater_than(self, db):
        db.create_node(["N"], {"v": 5})
        db.create_node(["N"], {"v": 15})
        result = list(db.execute("MATCH (n:N) WHERE n.v > 10 RETURN n.v"))
        assert result[0]["n.v"] == 15

    def test_greater_equal(self, db):
        db.create_node(["N"], {"v": 10})
        db.create_node(["N"], {"v": 5})
        result = list(db.execute("MATCH (n:N) WHERE n.v >= 10 RETURN n.v"))
        assert len(result) == 1


class TestLogicalOperators:
    """Logical: AND, OR, NOT, XOR."""

    def test_and(self, db):
        db.create_node(["N"], {"a": True, "b": True})
        db.create_node(["N"], {"a": True, "b": False})
        result = list(db.execute("MATCH (n:N) WHERE n.a AND n.b RETURN n"))
        assert len(result) == 1

    def test_or(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        db.create_node(["N"], {"v": 3})
        result = list(db.execute("MATCH (n:N) WHERE n.v = 1 OR n.v = 3 RETURN n.v"))
        assert len(result) == 2

    def test_not(self, db):
        db.create_node(["N"], {"v": 1})
        db.create_node(["N"], {"v": 2})
        result = list(db.execute("MATCH (n:N) WHERE NOT n.v = 1 RETURN n.v"))
        assert result[0]["n.v"] == 2

    def test_xor(self, db):
        db.create_node(["N"], {"a": True, "b": False})
        db.create_node(["N"], {"a": True, "b": True})
        db.create_node(["N"], {"a": False, "b": False})
        result = list(db.execute("MATCH (n:N) WHERE n.a XOR n.b RETURN n"))
        assert len(result) == 1


# =============================================================================
# String Predicates (sec 19)
# =============================================================================


class TestStringPredicates:
    """STARTS WITH, ENDS WITH, CONTAINS, LIKE, IN."""

    def test_starts_with(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute("MATCH (n:N) WHERE n.v STARTS WITH 'Am' RETURN n.v"))
        assert result[0]["n.v"] == "Amsterdam"

    def test_ends_with(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute("MATCH (n:N) WHERE n.v ENDS WITH 'in' RETURN n.v"))
        assert result[0]["n.v"] == "Berlin"

    def test_contains(self, db):
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute("MATCH (n:N) WHERE n.v CONTAINS 'ster' RETURN n.v"))
        assert result[0]["n.v"] == "Amsterdam"

    def test_like_percent(self, db):
        """LIKE with % wildcard."""
        db.create_node(["N"], {"v": "Amsterdam"})
        db.create_node(["N"], {"v": "Berlin"})
        result = list(db.execute("MATCH (n:N) WHERE n.v LIKE 'Am%' RETURN n.v"))
        assert result[0]["n.v"] == "Amsterdam"

    def test_like_underscore(self, db):
        """LIKE with _ single-char wildcard."""
        db.create_node(["N"], {"v": "ABC"})
        db.create_node(["N"], {"v": "AXC"})
        db.create_node(["N"], {"v": "ABCD"})
        result = list(db.execute("MATCH (n:N) WHERE n.v LIKE 'A_C' RETURN n.v"))
        vals = {r["n.v"] for r in result}
        assert vals == {"ABC", "AXC"}

    def test_in_list(self, db):
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        db.create_node(["N"], {"v": "c"})
        result = list(db.execute("MATCH (n:N) WHERE n.v IN ['a', 'c'] RETURN n.v"))
        assert len(result) == 2


# =============================================================================
# Literal Values (sec 20.3)
# =============================================================================


class TestLiterals:
    """Literal value expressions."""

    def test_null_literal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN null AS r"))
        assert result[0]["r"] is None

    def test_boolean_true(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN TRUE AS r"))
        assert result[0]["r"] is True

    def test_boolean_false(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN FALSE AS r"))
        assert result[0]["r"] is False

    def test_integer_decimal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN 42 AS r"))
        assert result[0]["r"] == 42

    def test_integer_hex(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN 0xFF AS r"))
        assert result[0]["r"] == 255

    def test_integer_octal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN 0o77 AS r"))
        assert result[0]["r"] == 63

    def test_float_decimal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN 3.14 AS r"))
        assert abs(result[0]["r"] - 3.14) < 0.001

    def test_float_scientific(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN 1.5e2 AS r"))
        assert result[0]["r"] == 150.0

    def test_string_single_quoted(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN 'hello' AS r"))
        assert result[0]["r"] == "hello"

    def test_string_double_quoted(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute('MATCH (n:N) RETURN "hello" AS r'))
        assert result[0]["r"] == "hello"

    def test_list_literal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN [1, 2, 3] AS r"))
        assert result[0]["r"] == [1, 2, 3]

    def test_map_literal(self, db):
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN {a: 1, b: 'x'} AS r"))
        assert result[0]["r"]["a"] == 1
        assert result[0]["r"]["b"] == "x"

    def test_date_literal(self, db):
        """DATE 'YYYY-MM-DD' typed temporal literal."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN DATE '2024-01-15' AS r"))
        assert "2024-01-15" in str(result[0]["r"])

    def test_time_literal(self, db):
        """TIME 'HH:MM:SS' typed temporal literal."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN TIME '14:30:00' AS r"))
        assert "14:30" in str(result[0]["r"])

    def test_datetime_literal(self, db):
        """DATETIME typed temporal literal."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN DATETIME '2024-01-15T14:30:00' AS r"))
        assert "2024" in str(result[0]["r"])

    def test_duration_literal(self, db):
        """DURATION typed temporal literal."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN DURATION 'P1Y2M' AS r"))
        assert result[0]["r"] is not None

    def test_zoned_datetime_literal(self, db):
        """ZONED DATETIME typed temporal literal."""
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute("MATCH (n:N) RETURN ZONED DATETIME '2024-01-15T14:30:00+05:30' AS r")
        )
        assert result[0]["r"] is not None

    def test_zoned_time_literal(self, db):
        """ZONED TIME typed temporal literal."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN ZONED TIME '14:30:00+05:30' AS r"))
        assert result[0]["r"] is not None


# =============================================================================
# Expression Forms (sec 20.7, 20.8)
# =============================================================================


class TestExpressionForms:
    """CASE, NULLIF, COALESCE, CAST, LET IN END, list comprehension, reduce."""

    def test_case_simple(self, db):
        """CASE expr WHEN val THEN result END."""
        db.create_node(["N"], {"v": "Amsterdam"})
        result = list(
            db.execute(
                "MATCH (n:N) "
                "RETURN CASE n.v WHEN 'Amsterdam' THEN 'NL' WHEN 'Berlin' THEN 'DE' END AS r"
            )
        )
        assert result[0]["r"] == "NL"

    def test_case_searched(self, db):
        """CASE WHEN cond THEN result ELSE default END."""
        db.create_node(["N"], {"v": 30})
        result = list(
            db.execute("MATCH (n:N) RETURN CASE WHEN n.v > 25 THEN 'high' ELSE 'low' END AS r")
        )
        assert result[0]["r"] == "high"

    def test_nullif(self, db):
        """NULLIF(a, b) returns null when a = b."""
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN NULLIF(n.v, 10) AS r"))
        assert result[0]["r"] is None

    def test_nullif_different(self, db):
        """NULLIF(a, b) returns a when a <> b."""
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN NULLIF(n.v, 20) AS r"))
        assert result[0]["r"] == 10

    def test_coalesce(self, db):
        """COALESCE returns first non-null."""
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN COALESCE(n.missing, n.v, 99) AS r"))
        assert result[0]["r"] == 10

    def test_cast_to_int(self, db):
        """CAST(expr AS INT)."""
        db.create_node(["N"], {"v": "42"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS INT) AS r"))
        assert result[0]["r"] == 42

    def test_cast_to_float(self, db):
        """CAST(expr AS FLOAT)."""
        db.create_node(["N"], {"v": "3.14"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS FLOAT) AS r"))
        assert abs(result[0]["r"] - 3.14) < 0.001

    def test_cast_to_string(self, db):
        """CAST(expr AS STRING)."""
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS STRING) AS r"))
        assert result[0]["r"] == "42"

    def test_cast_to_bool(self, db):
        """CAST(expr AS BOOL)."""
        db.create_node(["N"], {"v": "true"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS BOOLEAN) AS r"))
        assert result[0]["r"] is True

    def test_cast_to_date(self, db):
        """CAST(expr AS DATE)."""
        db.create_node(["N"], {"v": "2024-01-15"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS DATE) AS r"))
        assert "2024-01-15" in str(result[0]["r"])

    def test_cast_to_list(self, db):
        """CAST(expr AS LIST) wraps scalar in list."""
        db.create_node(["N"], {"v": 42})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS LIST) AS r"))
        assert result[0]["r"] == [42]

    def test_cast_to_zoned_datetime(self, db):
        """CAST(expr AS ZONED DATETIME)."""
        db.create_node(["N"], {"v": "2024-01-15T14:30:00+01:00"})
        result = list(db.execute("MATCH (n:N) RETURN CAST(n.v AS ZONED DATETIME) AS r"))
        assert result[0]["r"] is not None

    def test_let_in_expression(self, db):
        """LET ... IN ... END expression binding."""
        db.create_node(["N"], {"v": 5})
        result = list(db.execute("MATCH (n:N) RETURN LET x = n.v * 2 IN x + 1 END AS r"))
        assert result[0]["r"] == 11

    def test_list_comprehension(self, db):
        """[x IN list WHERE pred | transform] list comprehension."""
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute("MATCH (n:N) RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 2 | x * 10] AS r")
        )
        assert result[0]["r"] == [30, 40, 50]

    def test_list_comprehension_filter_only(self, db):
        """[x IN list WHERE pred] without transform."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN [x IN [1, 2, 3, 4] WHERE x > 2] AS r"))
        assert result[0]["r"] == [3, 4]

    def test_reduce(self, db):
        """reduce(acc = init, x IN list | expr) accumulator."""
        db.create_node(["N"], {"v": 1})
        result = list(
            db.execute("MATCH (n:N) RETURN reduce(acc = 0, x IN [1, 2, 3, 4] | acc + x) AS r")
        )
        assert result[0]["r"] == 10

    def test_property_access(self, db):
        """n.prop property access."""
        db.create_node(["N"], {"name": "Alix"})
        result = list(db.execute("MATCH (n:N) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_list_index_access(self, db):
        """list[index] access."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN [10, 20, 30][1] AS r"))
        assert result[0]["r"] == 20

    def test_session_user(self, db):
        """SESSION_USER returns current user."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN SESSION_USER AS r"))
        assert result[0]["r"] is not None


# =============================================================================
# Parameters (sec 20.4)
# =============================================================================


class TestParameters:
    """Dynamic parameters $param."""

    def test_param_in_where(self, db):
        """$param in WHERE clause."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        result = list(
            db.execute(
                "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
                {"name": "Alix"},
            )
        )
        assert result[0]["n.age"] == 30

    def test_param_in_return(self, db):
        """$param in RETURN expression."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN $val AS r", {"val": 42}))
        assert result[0]["r"] == 42

    def test_param_list(self, db):
        """$param as a list value."""
        db.create_node(["N"], {"v": "a"})
        db.create_node(["N"], {"v": "b"})
        db.create_node(["N"], {"v": "c"})
        result = list(
            db.execute(
                "MATCH (n:N) WHERE n.v IN $vals RETURN n.v",
                {"vals": ["a", "c"]},
            )
        )
        assert len(result) == 2


# =============================================================================
# Binary expressions in RETURN
# =============================================================================


class TestReturnExpressions:
    """Complex expressions in RETURN clause."""

    def test_comparison_in_return(self, db):
        """Boolean comparison in RETURN: n.v > 5."""
        db.create_node(["N"], {"v": 10})
        result = list(db.execute("MATCH (n:N) RETURN n.v > 5 AS r"))
        assert result[0]["r"] is True

    def test_arithmetic_in_return(self, db):
        """Arithmetic in RETURN: n.v + 10."""
        db.create_node(["N"], {"v": 30})
        result = list(db.execute("MATCH (n:N) RETURN n.v + 10 AS r"))
        assert result[0]["r"] == 40

    def test_aggregate_comparison_in_return(self, db):
        """count(n) > 0 in RETURN."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute("MATCH (n:N) RETURN count(n) > 0 AS r"))
        assert result[0]["r"] is True
