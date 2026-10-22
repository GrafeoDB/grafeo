"""Regression tests inspired by bugs found in other graph databases.

Covers MERGE semantics, pattern matching, aggregation edge cases,
Unicode handling, and query correctness.

Run: pytest tests/lpg/gql/test_regression_external.py -v
"""

import math

import pytest

# =============================================================================
# MERGE + UNWIND tuple count
# =============================================================================


class TestMergeUnwind:
    """MERGE inside UNWIND must produce one row per input."""

    def test_unwind_merge_returns_row_per_input(self, db):
        """UNWIND [1,1,1] + MERGE should produce 3 rows."""
        result = list(db.execute("UNWIND [1, 1, 1] AS i MERGE (:Item {val: i}) RETURN i"))
        assert len(result) == 3, "MERGE must emit one row per UNWIND input"

    def test_unwind_merge_creates_single_node(self, db):
        """Duplicate UNWIND values should create only one node."""
        db.execute("UNWIND [1, 1, 1] AS i MERGE (:Item {val: i})")
        result = list(db.execute("MATCH (n:Item) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_unwind_merge_distinct_creates_multiple(self, db):
        """Distinct UNWIND values each create a node."""
        db.execute("UNWIND [1, 2, 3] AS i MERGE (:Item {val: i})")
        result = list(db.execute("MATCH (n:Item) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_unwind_merge_mixed_create_and_match(self, db):
        """MERGE matches existing and creates new in one UNWIND."""
        db.execute("INSERT (:Item {val: 2})")
        db.execute("UNWIND [1, 2, 3] AS i MERGE (:Item {val: i})")
        result = list(db.execute("MATCH (n:Item) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_unwind_merge_on_create_on_match(self, db):
        """ON CREATE SET and ON MATCH SET apply correctly per-row."""
        db.execute("INSERT (:Item {val: 1, status: 'old'})")
        db.execute(
            "UNWIND [1, 2] AS i "
            "MERGE (n:Item {val: i}) "
            "ON CREATE SET n.status = 'new' "
            "ON MATCH SET n.status = 'updated'"
        )
        result = list(db.execute("MATCH (n:Item) RETURN n.val, n.status ORDER BY n.val"))
        assert len(result) == 2
        # val=1 existed: matched and updated
        assert result[0]["n.val"] == 1
        assert result[0]["n.status"] == "updated"
        # val=2 was new: created
        assert result[1]["n.val"] == 2
        assert result[1]["n.status"] == "new"

    def test_unwind_merge_identity_stable(self, db):
        """Same node returned each time MERGE matches it."""
        result = list(
            db.execute("UNWIND [1, 1, 1] AS i MERGE (n:Item {val: i}) RETURN id(n) AS nid")
        )
        ids = [r["nid"] for r in result]
        assert len(set(ids)) == 1, "All rows should reference the same node"


# =============================================================================
# MERGE with composite keys
# =============================================================================


class TestMergeCompositeKeys:
    """MERGE with multiple property keys must match on the full composite."""

    def test_identical_composite_no_duplicate(self, db):
        """Same two-property key should not create a second node."""
        db.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
        db.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
        result = list(db.execute("MATCH (n:City) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_partial_match_creates_new(self, db):
        """Same name, different country: must create a new node."""
        db.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
        db.execute("MERGE (:City {name: 'Amsterdam', country: 'US'})")
        result = list(db.execute("MATCH (n:City) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_three_property_composite(self, db):
        """Three-property composite key deduplication."""
        db.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Mitte'})")
        db.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Mitte'})")
        db.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Kreuzberg'})")
        result = list(db.execute("MATCH (n:Place) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2


# =============================================================================
# Relationship isomorphism
# =============================================================================


class TestRelationshipIsomorphism:
    """Same relationship must not be matched twice in a single pattern."""

    def test_single_edge_cannot_satisfy_two_hop(self, db):
        """One edge cannot be used as both r1 and r2 in a two-hop pattern."""
        db.execute("INSERT (:N {name: 'Alix'})-[:R]->(:N {name: 'Gus'})")
        result = list(db.execute("MATCH (a:N)-[r1:R]->(b:N)-[r2:R]->(c:N) RETURN count(*) AS cnt"))
        assert result[0]["cnt"] == 0


# =============================================================================
# OPTIONAL MATCH order independence
# =============================================================================


class TestOptionalMatchOrder:
    """Swapping OPTIONAL MATCH clauses must not change results."""

    def test_swapped_optional_match_same_results(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        db.execute("INSERT (:Person {name: 'Alix'})-[:WORKS_AT]->(:Company {name: 'Acme'})")
        r1 = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'}) "
                "OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person) "
                "OPTIONAL MATCH (p)-[:WORKS_AT]->(co:Company) "
                "RETURN p.name, friend.name, co.name"
            )
        )
        r2 = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'}) "
                "OPTIONAL MATCH (p)-[:WORKS_AT]->(co:Company) "
                "OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person) "
                "RETURN p.name, friend.name, co.name"
            )
        )
        assert len(r1) == len(r2), "Swapping OPTIONAL MATCH order must not change row count"


# =============================================================================
# Aggregation inside CALL subqueries
# =============================================================================


class TestAggregationInSubquery:
    """Aggregation functions inside CALL subqueries must work correctly."""

    def test_count_inside_call(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        result = list(db.execute("CALL { MATCH (n:Person) RETURN count(n) AS cnt } RETURN cnt"))
        assert len(result) == 1
        assert result[0]["cnt"] == 3


# =============================================================================
# COLLECT() completeness
# =============================================================================


class TestCollect:
    """COLLECT must gather all values without dropping any."""

    def test_collect_returns_all(self, db):
        for name in ["Vincent", "Alix", "Gus", "Jules", "Mia"]:
            db.execute(f"INSERT (:Person {{name: '{name}'}})")
        result = list(db.execute("MATCH (p:Person) RETURN collect(p.name) AS names"))
        assert len(result) == 1
        names = sorted(result[0]["names"])
        assert names == ["Alix", "Gus", "Jules", "Mia", "Vincent"]


# =============================================================================
# SUM overflow to infinity
# =============================================================================


class TestSumOverflow:
    """SUM of very large floats must return infinity, not error."""

    def test_sum_overflow(self, db):
        """Two f64::MAX values should overflow to +Infinity."""
        max_f64 = 1.7976931348623157e308
        result = list(
            db.execute(
                "UNWIND [$a, $b] AS val RETURN SUM(val) AS total",
                {"a": max_f64, "b": max_f64},
            )
        )
        assert len(result) == 1
        total = result[0]["total"]
        assert math.isinf(total) and total > 0, (
            f"SUM of two f64::MAX should be +Infinity, got {total}"
        )


# =============================================================================
# GROUP BY expression order independence
# =============================================================================


class TestGroupByExpressionOrder:
    """Column order in RETURN must not change GROUP BY results."""

    def test_column_order_does_not_affect_grouping(self, db):
        db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
        db.execute("INSERT (:Person {name: 'Gus', city: 'Amsterdam'})")
        db.execute("INSERT (:Person {name: 'Vincent', city: 'Berlin'})")

        r1 = list(
            db.execute("MATCH (p:Person) RETURN p.city AS city, count(p) AS cnt ORDER BY city")
        )
        r2 = list(
            db.execute("MATCH (p:Person) RETURN count(p) AS cnt, p.city AS city ORDER BY city")
        )
        cities1 = [r["city"] for r in r1]
        cities2 = [r["city"] for r in r2]
        assert cities1 == cities2


# =============================================================================
# WHERE filter must not corrupt projected values
# =============================================================================


class TestWhereFilterAndProjection:
    """WHERE must not replace returned property values with booleans."""

    def test_is_not_null_preserves_return_value(self, db):
        db.execute("INSERT (:Sensor {name: 'Temp', reading: 42.5})")
        db.execute("INSERT (:Sensor {name: 'Humidity'})")
        result = list(
            db.execute("MATCH (s:Sensor) WHERE s.reading IS NOT NULL RETURN s.name, s.reading")
        )
        assert len(result) == 1
        assert result[0]["s.name"] == "Temp"
        assert result[0]["s.reading"] == 42.5  # Must NOT be True


# =============================================================================
# Unicode and emoji in property values
# =============================================================================


class TestUnicodeEmoji:
    """Storing and reading back Unicode, emoji, and CJK characters."""

    def test_emoji_roundtrip(self, db):
        db.execute("INSERT (:Tag {symbol: '\U0001f389', name: 'party'})")
        result = list(db.execute("MATCH (t:Tag) RETURN t.symbol, t.name"))
        assert result[0]["t.symbol"] == "\U0001f389"  # 🎉

    def test_cjk_roundtrip(self, db):
        db.execute("INSERT (:City {name: '\u6771\u4eac'})")  # 東京
        result = list(db.execute("MATCH (c:City) RETURN c.name"))
        assert result[0]["c.name"] == "\u6771\u4eac"

    def test_combining_diacritics_roundtrip(self, db):
        """Combining diacritics (e + U+0301) must survive storage roundtrip."""
        # "calf" + e + combining acute accent (not the precomposed é)
        text = "calf\u0065\u0301"
        db.execute("INSERT (:Word {text: $text})", {"text": text})
        result = list(db.execute("MATCH (w:Word) RETURN w.text"))
        assert result[0]["w.text"] == text


# =============================================================================
# Self-loop pattern matching
# =============================================================================


class TestSelfLoop:
    """Self-referencing edges must be created and matched correctly."""

    def test_create_and_match_self_loop(self, db):
        db.execute("INSERT (a:Node {name: 'Alix'})-[:SELF]->(a)")
        result = list(db.execute("MATCH (a:Node)-[r:SELF]->(a) RETURN a.name"))
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"

    def test_self_loop_counted_once(self, db):
        db.execute("INSERT (a:Node {name: 'Alix'})-[:SELF]->(a)")
        result = list(db.execute("MATCH (:Node)-[r:SELF]->() RETURN count(r) AS cnt"))
        assert result[0]["cnt"] == 1


# =============================================================================
# Deleted node access in same session
# =============================================================================


class TestDeletedNodeAccess:
    """Deleted nodes must not be visible in subsequent queries."""

    def test_delete_then_match(self, db):
        db.execute("INSERT (:Temp {name: 'ephemeral'})")
        db.execute("MATCH (n:Temp) DELETE n")
        result = list(db.execute("MATCH (n:Temp) RETURN n.name"))
        assert len(result) == 0

    def test_detach_delete_clears_edges(self, db):
        db.execute("INSERT (:A {name: 'a'})-[:R]->(:B {name: 'b'})")
        db.execute("MATCH (a:A) DETACH DELETE a")
        result = list(db.execute("MATCH ()-[r:R]->() RETURN count(r) AS cnt"))
        assert result[0]["cnt"] == 0


# =============================================================================
# Nested map/list properties
# =============================================================================


class TestNestedProperties:
    """Deeply nested maps and heterogeneous lists must roundtrip correctly."""

    def test_nested_map_via_params(self, db):
        meta = {"b": {"c": 42}}
        db.execute("INSERT (:Data {meta: $meta})", {"meta": meta})
        result = list(db.execute("MATCH (d:Data) RETURN d.meta"))
        assert result[0]["d.meta"]["b"]["c"] == 42

    def test_heterogeneous_list_via_params(self, db):
        items = [1, "two", True, None]
        db.execute("INSERT (:Data {items: $items})", {"items": items})
        result = list(db.execute("MATCH (d:Data) RETURN d.items"))
        assert len(result[0]["d.items"]) == 4


# =============================================================================
# Constant folding in WHERE
# =============================================================================


class TestConstantFolding:
    """Constant WHERE clauses must not error."""

    def test_where_false_returns_empty(self, db):
        db.execute("INSERT (:N {val: 1})")
        result = list(db.execute("MATCH (n:N) WHERE 1 = 2 RETURN n"))
        assert len(result) == 0

    def test_where_true_returns_all(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 2})")
        result = list(db.execute("MATCH (n:N) WHERE 1 = 1 RETURN n.val"))
        assert len(result) == 2


# =============================================================================
# Cyclic graph traversal
# =============================================================================


class TestCyclicTraversal:
    """Variable-length paths through cycles must terminate."""

    def test_triangle_vle_terminates(self, db):
        db.execute(
            "INSERT (a:N {name: 'Alix'})-[:R]->(b:N {name: 'Gus'})"
            "-[:R]->(c:N {name: 'Vincent'})-[:R]->(a)"
        )
        result = list(db.execute("MATCH (a:N {name: 'Alix'})-[:R*1..5]->(b:N) RETURN b.name"))
        assert len(result) > 0, "Should find reachable nodes"
        assert len(result) < 100, f"Should not explode: got {len(result)} rows"


# =============================================================================
# MERGE with NULL node reference
# =============================================================================


class TestMergeNullReference:
    """MERGE on a NULL variable (from OPTIONAL MATCH) should error."""

    def test_merge_rel_with_null_source_errors(self, db):
        """OPTIONAL MATCH that matches nothing + MERGE should fail."""
        with pytest.raises(RuntimeError):
            db.execute("OPTIONAL MATCH (n:NonExistent) MERGE (n)-[:R]->(m:Target {name: 'Alix'})")

    def test_merge_rel_with_null_target_errors(self, db):
        """NULL target in MERGE relationship should fail."""
        db.execute("INSERT (:Source {name: 'Gus'})")
        with pytest.raises(RuntimeError):
            db.execute(
                "MATCH (a:Source {name: 'Gus'}) OPTIONAL MATCH (b:NonExistent) MERGE (a)-[:R]->(b)"
            )

    def test_standalone_merge_unaffected(self, db):
        """Standalone MERGE (no OPTIONAL MATCH) should still work."""
        db.execute("MERGE (:Person {name: 'Mia'})")
        result = list(db.execute("MATCH (n:Person {name: 'Mia'}) RETURN n.name"))
        assert len(result) == 1


# =============================================================================
# labels() / type() in aggregation and ORDER BY (#187)
# =============================================================================


class TestIssue187LabelsTypeAggregation:
    """labels() and type() must work correctly in GROUP BY and ORDER BY."""

    def test_labels_group_by_count(self, db):
        """GROUP BY labels(n)[0] with COUNT should produce correct groups."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(
            db.execute("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label")
        )
        assert len(result) == 2
        assert result[0]["label"] == "City"
        assert result[0]["cnt"] == 1
        assert result[1]["label"] == "Person"
        assert result[1]["cnt"] == 2

    def test_type_group_by_count(self, db):
        """GROUP BY type(r) with COUNT should produce correct groups."""
        alix = db.create_node(["Person"], {"name": "Alix"})
        gus = db.create_node(["Person"], {"name": "Gus"})
        acme = db.create_node(["Company"], {"name": "Acme"})
        db.create_edge(alix.id, gus.id, "KNOWS")
        db.create_edge(alix.id, acme.id, "WORKS_AT")
        db.create_edge(gus.id, acme.id, "WORKS_AT")
        result = list(
            db.execute("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY t")
        )
        assert len(result) == 2
        assert result[0]["t"] == "KNOWS"
        assert result[0]["cnt"] == 1
        assert result[1]["t"] == "WORKS_AT"
        assert result[1]["cnt"] == 2

    def test_labels_order_by(self, db):
        """ORDER BY labels(n)[0] sorts by first label alphabetically."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["City"], {"name": "Amsterdam"})
        result = list(db.execute("MATCH (n) RETURN n.name ORDER BY labels(n)[0]"))
        assert len(result) == 2
        # "City" < "Person" alphabetically
        assert result[0]["n.name"] == "Amsterdam"
        assert result[1]["n.name"] == "Alix"

    def test_type_order_by(self, db):
        """ORDER BY type(r) sorts by edge type alphabetically."""
        alix = db.create_node(["Person"], {"name": "Alix"})
        gus = db.create_node(["Person"], {"name": "Gus"})
        acme = db.create_node(["Company"], {"name": "Acme"})
        db.create_edge(alix.id, gus.id, "KNOWS")
        db.create_edge(alix.id, acme.id, "WORKS_AT")
        result = list(db.execute("MATCH ()-[r]->() RETURN type(r) AS t ORDER BY t"))
        assert len(result) == 2
        assert result[0]["t"] == "KNOWS"
        assert result[1]["t"] == "WORKS_AT"

    def test_labels_group_by_sum(self, db):
        """GROUP BY labels(n)[0] with SUM on a numeric property."""
        db.create_node(["Person"], {"name": "Alix", "val": 10})
        db.create_node(["Person"], {"name": "Gus", "val": 20})
        db.create_node(["City"], {"name": "Amsterdam", "val": 5})
        result = list(
            db.execute("MATCH (n) RETURN labels(n)[0] AS label, sum(n.val) AS total ORDER BY label")
        )
        assert len(result) == 2
        assert result[0]["label"] == "City"
        assert result[0]["total"] == 5
        assert result[1]["label"] == "Person"
        assert result[1]["total"] == 30

    def test_combined_group_by_and_order_by(self, db):
        """Both GROUP BY and ORDER BY use labels(), descending."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Vincent"})
        db.create_node(["City"], {"name": "Amsterdam"})
        db.create_node(["City"], {"name": "Berlin"})
        result = list(
            db.execute(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY labels(n)[0] DESC"
            )
        )
        assert len(result) == 2
        assert result[0]["label"] == "Person"
        assert result[0]["cnt"] == 3
        assert result[1]["label"] == "City"
        assert result[1]["cnt"] == 2


# =============================================================================
# Integer overflow detection
# Inspired by Neo4j #13674
# =============================================================================


class TestIntegerOverflow:
    """Arithmetic overflow must not silently wrap."""

    @pytest.mark.skip(
        reason="panics on overflow instead of returning error: needs checked arithmetic"
    )
    def test_i64_max_plus_one(self, db):
        """i64::MAX + 1 should error or not silently wrap."""
        try:
            result = list(db.execute("RETURN 9223372036854775807 + 1 AS r"))
            # If it returns, must not be i64::MIN
            assert result[0]["r"] != -9223372036854775808
        except RuntimeError:
            pass  # Overflow error is acceptable

    def test_i64_min_minus_one(self, db):
        """i64::MIN - 1 should error or not silently wrap."""
        try:
            result = list(db.execute("RETURN -9223372036854775808 - 1 AS r"))
            assert result[0]["r"] != 9223372036854775807
        except RuntimeError:
            pass  # Overflow error is acceptable


# =============================================================================
# Variable-length path enumeration
# Inspired by FalkorDB #1450
# =============================================================================


class TestVariableLengthPathEnumeration:
    """Variable-length paths must enumerate all distinct paths, not just endpoints."""

    def test_diamond_two_hop_enumerates_all_paths(self, db):
        """Diamond graph: A->B1->C, A->B2->C should yield 2 two-hop paths."""
        db.execute(
            "INSERT (a:N {name: 'Alix'})-[:R]->(b1:N {name: 'Gus'}), "
            "(a)-[:R]->(b2:N {name: 'Vincent'}), "
            "(b1)-[:R]->(c:N {name: 'Jules'}), "
            "(b2)-[:R]->(c)"
        )
        result = list(
            db.execute("MATCH (:N {name: 'Alix'})-[:R*2]->(c:N {name: 'Jules'}) RETURN c.name")
        )
        assert len(result) == 2, f"Diamond should yield 2 paths, got {len(result)}"

    def test_variable_length_respects_max_hops(self, db):
        """*1..2 from chain start should reach only 2 nodes."""
        db.execute(
            "INSERT (:N {name: 'a'})-[:R]->(:N {name: 'b'})-[:R]->"
            "(:N {name: 'c'})-[:R]->(:N {name: 'd'})"
        )
        result = list(
            db.execute("MATCH (:N {name: 'a'})-[:R*1..2]->(b) RETURN b.name ORDER BY b.name")
        )
        assert len(result) == 2


# =============================================================================
# UNWIND NULL produces zero rows
# Inspired by FalkorDB #1031
# =============================================================================


class TestUnwindNull:
    """UNWIND NULL and UNWIND [] must produce zero rows."""

    def test_unwind_null_zero_rows(self, db):
        result = list(db.execute("UNWIND NULL AS x RETURN x"))
        assert len(result) == 0

    def test_unwind_empty_list_zero_rows(self, db):
        result = list(db.execute("UNWIND [] AS x RETURN x"))
        assert len(result) == 0

    def test_unwind_null_no_writes(self, db):
        """UNWIND NULL should not execute downstream INSERTs."""
        try:
            db.execute("UNWIND NULL AS x INSERT (:Ghost {val: x})")
        except RuntimeError:
            pass  # Error on UNWIND NULL is acceptable
        result = list(db.execute("MATCH (n:Ghost) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 0


# =============================================================================
# NULL predicate semantics
# Inspired by Neo4j #13727
# =============================================================================


class TestNullPredicateSemantics:
    """NULL comparison behavior in WHERE clauses."""

    def test_equality_with_null_returns_empty(self, db):
        db.execute("INSERT (:N {val: 1})")
        result = list(db.execute("MATCH (n:N) WHERE 1 = NULL RETURN n"))
        assert len(result) == 0

    def test_in_empty_list_returns_empty(self, db):
        db.execute("INSERT (:N {val: 1})")
        result = list(db.execute("MATCH (n:N) WHERE 1 IN [] RETURN n"))
        assert len(result) == 0

    def test_null_not_equal_to_null(self, db):
        db.execute("INSERT (:N {val: 1})")
        result = list(db.execute("MATCH (n:N) WHERE NULL = NULL RETURN n"))
        assert len(result) == 0

    def test_null_is_null(self, db):
        db.execute("INSERT (:N {val: 1})")
        result = list(db.execute("MATCH (n:N) WHERE NULL IS NULL RETURN n.val"))
        assert len(result) == 1


# =============================================================================
# Predicate rewriting: NOT((a = b) IS NULL)
# Inspired by Neo4j #13642
# =============================================================================


class TestPredicateRewriting:
    """NOT((comparison) IS NULL) must not confuse optimizer."""

    def test_not_is_null_on_comparison(self, db):
        db.execute("INSERT (:X {k1: 100})-[:R]->(:Y {k2: 34})")
        result = list(
            db.execute(
                "MATCH (x:X)-[:R]->(y:Y) WHERE NOT ((x.k1 = y.k2) IS NULL) RETURN x.k1, y.k2"
            )
        )
        assert len(result) == 1, "NOT((100=34) IS NULL) is TRUE, row must pass"

    def test_not_is_null_with_null_property(self, db):
        db.execute("INSERT (:X {k1: 100})-[:R]->(:Y {name: 'test'})")
        result = list(
            db.execute("MATCH (x:X)-[:R]->(y:Y) WHERE NOT ((x.k1 = y.k2) IS NULL) RETURN x.k1")
        )
        assert len(result) == 0, "NOT((100=NULL) IS NULL) is FALSE"


# =============================================================================
# Double delete idempotency
# Inspired by FalkorDB #1018
# =============================================================================


class TestDoubleDelete:
    """Deleting an already-deleted entity must not corrupt state."""

    def test_delete_then_count_zero(self, db):
        db.execute("INSERT (:Temp {name: 'Alix'})")
        db.execute("MATCH (n:Temp) DELETE n")
        try:
            db.execute("MATCH (n:Temp) DELETE n")
        except RuntimeError:
            pass  # Double-delete may error, which is acceptable
        result = list(db.execute("MATCH (n) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 0


# =============================================================================
# OPTIONAL MATCH with Cartesian product
# Inspired by FalkorDB #1281
# =============================================================================


class TestOptionalMatchCartesian:
    """OPTIONAL MATCH with rebound variables must not crash."""

    def test_optional_match_no_match_produces_nulls(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})")
        result = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'}) "
                "OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
                "RETURN p.name, c.name"
            )
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Alix"
        assert result[0]["c.name"] is None


# =============================================================================
# WHERE filter on traversal endpoints
# Inspired by LadybugDB #273
# =============================================================================


class TestWhereFilterOnTraversal:
    """Filters on destination/edge properties in multi-hop queries."""

    def test_filter_on_destination_property(self, db):
        db.execute(
            "INSERT (:Person {name: 'Alix'})-[:LIVES_IN]->(:City {name: 'Amsterdam', pop: 900000})"
        )
        db.execute(
            "INSERT (:Person {name: 'Gus'})-[:LIVES_IN]->(:City {name: 'Berlin', pop: 3700000})"
        )
        result = list(
            db.execute(
                "MATCH (p:Person)-[:LIVES_IN]->(c:City) WHERE c.pop > 1000000 RETURN p.name, c.name"
            )
        )
        assert len(result) == 1
        assert result[0]["p.name"] == "Gus"

    def test_filter_on_edge_property(self, db):
        db.execute(
            "INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})"
        )
        db.execute(
            "INSERT (:Person {name: 'Vincent'})-[:KNOWS {since: 2024}]->(:Person {name: 'Jules'})"
        )
        result = list(
            db.execute(
                "MATCH (a:Person)-[k:KNOWS]->(b:Person) WHERE k.since >= 2023 RETURN a.name, b.name"
            )
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Vincent"


# =============================================================================
# Empty aggregation
# =============================================================================


class TestEmptyAggregation:
    """Aggregation functions over zero rows."""

    def test_count_on_empty(self, db):
        result = list(db.execute("MATCH (n:NonExistent) RETURN count(n) AS cnt"))
        assert len(result) == 1
        assert result[0]["cnt"] == 0

    def test_avg_on_empty_returns_null(self, db):
        result = list(db.execute("MATCH (n:NonExistent) RETURN avg(n.val) AS a"))
        assert len(result) == 1
        assert result[0]["a"] is None

    def test_min_max_on_empty_returns_null(self, db):
        result = list(db.execute("MATCH (n:NonExistent) RETURN min(n.val) AS lo, max(n.val) AS hi"))
        assert len(result) == 1
        assert result[0]["lo"] is None
        assert result[0]["hi"] is None


# =============================================================================
# Relationship direction
# =============================================================================


class TestRelationshipDirection:
    """Arrow direction must be respected in pattern matching."""

    def test_backward_arrow(self, db):
        db.execute("INSERT (:A {name: 'Alix'})-[:FOLLOWS]->(:B {name: 'Gus'})")
        fwd = list(db.execute("MATCH (a:A)-[:FOLLOWS]->(b:B) RETURN a.name"))
        bwd = list(db.execute("MATCH (b:B)<-[:FOLLOWS]-(a:A) RETURN a.name"))
        wrong = list(db.execute("MATCH (a:A)<-[:FOLLOWS]-(b:B) RETURN a.name"))
        assert len(fwd) == 1
        assert len(bwd) == 1
        assert len(wrong) == 0

    def test_undirected_both_directions(self, db):
        db.execute("INSERT (:N {name: 'Alix'})-[:KNOWS]->(:N {name: 'Gus'})")
        result = list(db.execute("MATCH (a:N)-[:KNOWS]-(b:N) RETURN a.name ORDER BY a.name"))
        assert len(result) == 2


# =============================================================================
# MERGE edge patterns
# =============================================================================


class TestMergeEdgePatterns:
    """MERGE on relationships: create when missing, match when present."""

    def test_merge_creates_relationship(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})")
        db.execute("INSERT (:Person {name: 'Gus'})")
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)-[:KNOWS]->(b)"
        )
        result = list(db.execute("MATCH ()-[k:KNOWS]->() RETURN count(k) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_merge_does_not_duplicate_relationship(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)-[:KNOWS]->(b)"
        )
        result = list(db.execute("MATCH ()-[k:KNOWS]->() RETURN count(k) AS cnt"))
        assert result[0]["cnt"] == 1


# =============================================================================
# Delete + re-insert consistency
# Inspired by LadybugDB #180
# =============================================================================


class TestDeleteReinsert:
    """Data re-inserted after deletion must be fully visible."""

    def test_delete_all_reinsert(self, db):
        db.execute("INSERT (:Item {val: 1})")
        db.execute("INSERT (:Item {val: 2})")
        db.execute("MATCH (n:Item) DELETE n")
        db.execute("INSERT (:Item {val: 10})")
        db.execute("INSERT (:Item {val: 20})")
        result = list(db.execute("MATCH (n:Item) RETURN n.val ORDER BY n.val"))
        assert len(result) == 2
        assert result[0]["n.val"] == 10
        assert result[1]["n.val"] == 20


# =============================================================================
# Multi-label operations
# =============================================================================


class TestMultiLabel:
    """Nodes with multiple labels must be matchable by any of them."""

    def test_match_by_any_label(self, db):
        db.execute("INSERT (:Person:Employee {name: 'Alix'})")
        r1 = list(db.execute("MATCH (n:Person) RETURN n.name"))
        r2 = list(db.execute("MATCH (n:Employee) RETURN n.name"))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_labels_returns_all(self, db):
        db.execute("INSERT (:Person:Employee:Manager {name: 'Alix'})")
        result = list(db.execute("MATCH (n:Person) RETURN labels(n) AS lbls"))
        labels = sorted(result[0]["lbls"])
        assert labels == ["Employee", "Manager", "Person"]


# =============================================================================
# Property type edge cases
# =============================================================================


class TestPropertyTypeEdgeCases:
    """Falsy values (0, false, '') are valid, not NULL."""

    def test_integer_float_comparison(self, db):
        db.execute("INSERT (:N {ival: 42, fval: 42.0})")
        result = list(db.execute("MATCH (n:N) WHERE n.ival = n.fval RETURN n.ival"))
        assert len(result) == 1

    def test_missing_property_is_null(self, db):
        db.execute("INSERT (:N {name: 'Alix'})")
        result = list(db.execute("MATCH (n:N) RETURN n.nonexistent AS val"))
        assert result[0]["val"] is None

    def test_empty_string_is_not_null(self, db):
        db.execute("INSERT (:N {name: ''})")
        result = list(db.execute("MATCH (n:N) WHERE n.name IS NOT NULL RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == ""

    def test_zero_is_not_null(self, db):
        db.execute("INSERT (:N {val: 0})")
        result = list(db.execute("MATCH (n:N) WHERE n.val IS NOT NULL RETURN n.val"))
        assert len(result) == 1
        assert result[0]["n.val"] == 0

    def test_false_is_not_null(self, db):
        db.execute("INSERT (:N {flag: false})")
        result = list(db.execute("MATCH (n:N) WHERE n.flag IS NOT NULL RETURN n.flag"))
        assert len(result) == 1
        assert result[0]["n.flag"] is False


# =============================================================================
# AVG large integers
# Inspired by ArangoDB #21096
# =============================================================================


class TestAvgLargeIntegers:
    """AVG of large integers must not silently overflow."""

    def test_avg_identical_large_values(self, db):
        db.execute("INSERT (:M {val: 389916982198384})")
        db.execute("INSERT (:M {val: 389916982198384})")
        db.execute("INSERT (:M {val: 389916982198384})")
        result = list(db.execute("MATCH (m:M) RETURN avg(m.val) AS a"))
        assert abs(result[0]["a"] - 389916982198384) < 1.0


# =============================================================================
# List comparison
# Inspired by ArangoDB #2477
# =============================================================================


class TestListComparison:
    """List equality edge cases."""

    def test_empty_list_equals_empty_list(self, db):
        result = list(db.execute("RETURN [] = [] AS eq"))
        assert result[0]["eq"] is True

    def test_list_same_elements(self, db):
        result = list(db.execute("RETURN [1, 2, 3] = [1, 2, 3] AS eq"))
        assert result[0]["eq"] is True

    def test_list_different_elements(self, db):
        result = list(db.execute("RETURN [1, 2] = [1, 3] AS eq"))
        assert result[0]["eq"] is False

    def test_empty_list_not_equal_list_with_null(self, db):
        result = list(db.execute("RETURN [] = [NULL] AS eq"))
        assert result[0]["eq"] is not True


# =============================================================================
# OPTIONAL MATCH + aggregation scope
# Inspired by Memgraph #3970
# =============================================================================


class TestOptionalMatchAggregation:
    """Outer variable must survive OPTIONAL MATCH + aggregation."""

    def test_count_preserves_all_rows(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})-[:FRIEND]->(:Person {name: 'Gus'})")
        db.execute("INSERT (:Person {name: 'Vincent'})")
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "OPTIONAL MATCH (p)-[:FRIEND]->(f:Person) "
                "RETURN p.name, count(f) AS fc "
                "ORDER BY p.name"
            )
        )
        assert len(result) == 3
        assert result[0]["p.name"] == "Alix"
        assert result[0]["fc"] == 1
        assert result[2]["p.name"] == "Vincent"
        assert result[2]["fc"] == 0


# =============================================================================
# UNION correctness
# Inspired by Memgraph #3909
# =============================================================================


class TestUnionCorrectness:
    """Basic UNION and UNION ALL semantics."""

    def test_union_deduplicates(self, db):
        result = list(db.execute("RETURN 1 AS x UNION RETURN 1 AS x"))
        assert len(result) == 1

    def test_union_all_preserves_duplicates(self, db):
        result = list(db.execute("RETURN 1 AS x UNION ALL RETURN 1 AS x"))
        assert len(result) == 2

    def test_union_different_values(self, db):
        result = list(db.execute("RETURN 1 AS x UNION RETURN 2 AS x"))
        assert len(result) == 2


# =============================================================================
# OR condition correctness
# Inspired by JanusGraph #4786
# =============================================================================


class TestOrCondition:
    """OR in WHERE must match either branch."""

    def test_or_matches_both(self, db):
        db.execute("INSERT (:Item {prop: 'A'})")
        db.execute("INSERT (:Item {prop: 'B'})")
        db.execute("INSERT (:Item {prop: 'C'})")
        result = list(
            db.execute(
                "MATCH (n:Item) WHERE n.prop = 'A' OR n.prop = 'B' RETURN n.prop ORDER BY n.prop"
            )
        )
        assert len(result) == 2
        assert result[0]["n.prop"] == "A"
        assert result[1]["n.prop"] == "B"

    def test_or_with_and_precedence(self, db):
        db.execute("INSERT (:Item {a: 1, b: 2})")
        db.execute("INSERT (:Item {a: 3, b: 4})")
        db.execute("INSERT (:Item {a: 5, b: 6})")
        result = list(
            db.execute(
                "MATCH (n:Item) "
                "WHERE (n.a = 1 AND n.b = 2) OR (n.a = 5 AND n.b = 6) "
                "RETURN n.a ORDER BY n.a"
            )
        )
        assert len(result) == 2
        assert result[0]["n.a"] == 1
        assert result[1]["n.a"] == 5

    def test_not_inverts_filter(self, db):
        db.execute("INSERT (:Item {val: 1})")
        db.execute("INSERT (:Item {val: 2})")
        db.execute("INSERT (:Item {val: 3})")
        result = list(db.execute("MATCH (n:Item) WHERE NOT n.val = 2 RETURN n.val ORDER BY n.val"))
        assert len(result) == 2
        assert result[0]["n.val"] == 1
        assert result[1]["n.val"] == 3


# =============================================================================
# Type coercion: string 'false' vs boolean false
# Inspired by JanusGraph #4220
# =============================================================================


class TestTypeCoercionStringBool:
    """String 'false'/'true' must not equal boolean false/true."""

    def test_string_false_not_equal_bool_false(self, db):
        db.execute("INSERT (:N {val: false})")
        db.execute("INSERT (:N {val: 'false'})")
        result = list(db.execute("MATCH (n:N) WHERE n.val <> 'false' RETURN n.val"))
        assert len(result) == 1
        assert result[0]["n.val"] is False

    def test_string_true_not_equal_bool_true(self, db):
        db.execute("INSERT (:N {val: true})")
        db.execute("INSERT (:N {val: 'true'})")
        result = list(db.execute("MATCH (n:N) WHERE n.val <> 'true' RETURN n.val"))
        assert len(result) == 1
        assert result[0]["n.val"] is True


# =============================================================================
# Self-loop with variable-length expansion
# Inspired by Kuzu #5989
# =============================================================================


class TestSelfLoopVariableLength:
    """Variable-length expansion must terminate with self-loops."""

    def test_self_loop_varlength_terminates(self, db):
        db.execute("INSERT (a:N {name: 'Alix'})-[:LOOP]->(a)")
        db.execute(
            "MATCH (a:N {name: 'Alix'}) "
            "INSERT (a)-[:R]->(b:N {name: 'Gus'})-[:R]->(c:N {name: 'Vincent'})"
        )
        result = list(db.execute("MATCH (:N {name: 'Alix'})-[:R*1..3]->(b) RETURN b.name"))
        assert isinstance(result, list)  # Must terminate, not hang


# =============================================================================
# IN operator correctness
# Inspired by Kuzu #6010
# =============================================================================


class TestInOperator:
    """IN list membership operator."""

    def test_in_found(self, db):
        result = list(db.execute("RETURN 2 IN [1, 2, 3] AS found"))
        assert result[0]["found"] is True

    def test_in_not_found(self, db):
        result = list(db.execute("RETURN 4 IN [1, 2, 3] AS found"))
        assert result[0]["found"] is False

    def test_in_empty_list(self, db):
        result = list(db.execute("RETURN 1 IN [] AS found"))
        assert result[0]["found"] is False

    def test_in_with_null_found(self, db):
        result = list(db.execute("RETURN 1 IN [1, NULL] AS found"))
        assert result[0]["found"] is True

    def test_in_with_null_not_found(self, db):
        result = list(db.execute("RETURN 2 IN [1, NULL] AS found"))
        assert result[0]["found"] is None  # UNKNOWN


# =============================================================================
# ORDER BY with NULL properties
# Inspired by JanusGraph #3269
# =============================================================================


class TestOrderByNulls:
    """NULL properties in ORDER BY should sort to end and include all rows."""

    def test_order_by_partial_property(self, db):
        db.execute("INSERT (:N {name: 'Alix', score: 90})")
        db.execute("INSERT (:N {name: 'Gus'})")  # no score
        db.execute("INSERT (:N {name: 'Vincent', score: 80})")
        result = list(db.execute("MATCH (n:N) RETURN n.name, n.score ORDER BY n.score"))
        assert len(result) == 3
        assert result[2]["n.score"] is None


# =============================================================================
# MERGE after delete
# Inspired by Memgraph #2093
# =============================================================================


class TestMergeAfterDelete:
    """MERGE must not match a node deleted earlier in the session."""

    def test_merge_after_delete_creates_new(self, db):
        db.execute("INSERT (:Singleton {key: 'only'})")
        db.execute("MATCH (n:Singleton) DELETE n")
        db.execute("MERGE (:Singleton {key: 'only'})")
        result = list(db.execute("MATCH (n:Singleton) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1


# =============================================================================
# Label intersection across MATCH clauses
# Inspired by Memgraph #2875
# =============================================================================


class TestLabelIntersection:
    """Same variable in multiple MATCH clauses must intersect labels."""

    @pytest.mark.skip(reason="label constraints not intersected across MATCH clauses")
    def test_label_intersection_filters(self, db):
        db.execute("INSERT (:A:B {name: 'both'})")
        db.execute("INSERT (:A {name: 'only_a'})")
        db.execute("INSERT (:B {name: 'only_b'})")
        result = list(db.execute("MATCH (n:A) MATCH (n:B) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "both"


# =============================================================================
# IS NOT NULL operator precedence
# Inspired by Memgraph #2457
# =============================================================================


class TestIsNotNullPrecedence:
    """(expr) IS NOT NULL must evaluate correctly."""

    def test_boolean_expr_is_not_null(self, db):
        db.execute("INSERT (:N {val: 1})")
        result = list(db.execute("MATCH (n:N) WHERE (1 = 1) IS NOT NULL RETURN n.val"))
        assert len(result) == 1


# =============================================================================
# Quantifier functions on empty list
# Inspired by Memgraph #2481
# =============================================================================


class TestQuantifierFunctions:
    """any/all/none on empty list have spec-defined results."""

    def test_any_empty_is_false(self, db):
        result = list(db.execute("RETURN any(x IN [] WHERE x > 0) AS r"))
        assert result[0]["r"] is False

    def test_all_empty_is_true(self, db):
        result = list(db.execute("RETURN all(x IN [] WHERE x > 0) AS r"))
        assert result[0]["r"] is True

    def test_none_empty_is_true(self, db):
        result = list(db.execute("RETURN none(x IN [] WHERE x > 0) AS r"))
        assert result[0]["r"] is True


# =============================================================================
# String escapes
# Inspired by Kuzu #5814
# =============================================================================


class TestStringEscapes:
    """Escape sequences in string literals must be stored correctly."""

    def test_newline_escape(self, db):
        db.execute('INSERT (:Entry {text: "line1\\nline2"})')
        result = list(db.execute("MATCH (e:Entry) RETURN e.text"))
        assert "\n" in result[0]["e.text"]

    def test_tab_escape(self, db):
        db.execute('INSERT (:Entry {text: "col1\\tcol2"})')
        result = list(db.execute("MATCH (e:Entry) RETURN e.text"))
        assert "\t" in result[0]["e.text"]


# =============================================================================
# NULL join semantics
# Inspired by Kuzu #5893
# =============================================================================


class TestNullJoinSemantics:
    """NULL = NULL must not match in WHERE clause equality."""

    def test_null_equality_in_cross_match(self, db):
        db.execute("INSERT (:Account {svc: 'A', name: 'Alix'})")
        db.execute("INSERT (:Account {svc: 'B', name: 'Alix'})")
        db.execute("INSERT (:Account {svc: 'A'})")  # no name
        db.execute("INSERT (:Account {svc: 'B'})")  # no name
        result = list(
            db.execute(
                "MATCH (a:Account {svc: 'A'}) "
                "MATCH (b:Account {svc: 'B'}) "
                "WHERE a.name = b.name "
                "RETURN a.name"
            )
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "Alix"


# =============================================================================
# NULL function arguments
# Inspired by Kuzu #5959
# =============================================================================


class TestNullFunctionArguments:
    """Functions on NULL entity must return NULL, not crash."""

    def test_type_of_null_relationship(self, db):
        db.execute("INSERT (:N {name: 'Alix'})")
        result = list(
            db.execute("MATCH (n:N) OPTIONAL MATCH (n)-[r:NONEXISTENT]->() RETURN type(r) AS t")
        )
        assert result[0]["t"] is None


# =============================================================================
# Delete + re-create edge
# Inspired by Dgraph #9422
# =============================================================================


class TestDeleteRecreateEdge:
    """Replacing an edge: old must be gone, new must be visible."""

    def test_replace_edge(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})-[:LIKES]->(:Fruit {name: 'apple'})")
        db.execute("INSERT (:Fruit {name: 'banana'})")
        db.execute("MATCH (:Person {name: 'Alix'})-[r:LIKES]->() DELETE r")
        db.execute(
            "MATCH (p:Person {name: 'Alix'}), (f:Fruit {name: 'banana'}) INSERT (p)-[:LIKES]->(f)"
        )
        result = list(
            db.execute("MATCH (:Person {name: 'Alix'})-[:LIKES]->(f:Fruit) RETURN f.name")
        )
        assert len(result) == 1
        assert result[0]["f.name"] == "banana"


# =============================================================================
# LIMIT with ORDER BY
# Inspired by Dgraph #9239
# =============================================================================


class TestLimitWithOrder:
    """LIMIT must be respected with ORDER BY."""

    def test_limit_with_order_by(self, db):
        for i in range(20):
            db.execute(f"INSERT (:Item {{seq: {i}}})")
        result = list(db.execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 5"))
        assert len(result) == 5
        assert result[0]["n.seq"] == 0
        assert result[4]["n.seq"] == 4

    def test_limit_larger_than_result(self, db):
        db.execute("INSERT (:Item {seq: 1})")
        db.execute("INSERT (:Item {seq: 2})")
        result = list(db.execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 100"))
        assert len(result) == 2


# =============================================================================
# Idempotent property SET
# Inspired by Dgraph #9519
# =============================================================================


class TestIdempotentSet:
    """SET property to same value must not break other updates."""

    def test_set_same_value_with_other_change(self, db):
        db.execute("INSERT (:N {code: 'X', note: 'initial'})")
        db.execute("MATCH (n:N {code: 'X'}) SET n.code = 'X', n.note = 'updated'")
        result = list(db.execute("MATCH (n:N {code: 'X'}) RETURN n.note"))
        assert result[0]["n.note"] == "updated"


# =============================================================================
# Float precision
# Inspired by Dgraph #9491
# =============================================================================


class TestFloatPrecision:
    """Full float64 precision must be preserved."""

    def test_float64_roundtrip(self, db):
        db.execute("INSERT (:M {val: 0.123456789012345})")
        result = list(db.execute("MATCH (m:M) RETURN m.val"))
        assert abs(result[0]["m.val"] - 0.123456789012345) < 1e-15

    def test_repeated_edge_updates_no_stale(self, db):
        db.execute("INSERT (:Host {name: 'h1'})-[:NESTS]->(:Nest {name: 'n1'})")
        db.execute("INSERT (:Nest {name: 'n2'})")
        db.execute("INSERT (:Nest {name: 'n3'})")
        db.execute("MATCH (:Host {name: 'h1'})-[r:NESTS]->() DELETE r")
        db.execute("MATCH (h:Host {name: 'h1'}), (n:Nest {name: 'n2'}) INSERT (h)-[:NESTS]->(n)")
        db.execute("MATCH (:Host {name: 'h1'})-[r:NESTS]->() DELETE r")
        db.execute("MATCH (h:Host {name: 'h1'}), (n:Nest {name: 'n3'}) INSERT (h)-[:NESTS]->(n)")
        result = list(db.execute("MATCH (:Host {name: 'h1'})-[:NESTS]->(n) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "n3"


# =============================================================================
# NULL grouping key
# Inspired by ArangoDB #14672
# =============================================================================


class TestNullGroupingKey:
    """NULL in one GROUP BY column must not collapse distinct values in another."""

    def test_null_column_no_collapse(self, db):
        db.execute("INSERT (:Q {status: 'ok'})")
        db.execute("INSERT (:Q {status: 'ok'})")
        db.execute("INSERT (:Q {status: 'reject'})")
        result = list(
            db.execute(
                "MATCH (q:Q) RETURN q.status AS status, q.extra AS extra, count(*) AS cnt "
                "ORDER BY status"
            )
        )
        assert len(result) >= 2


# =============================================================================
# Inequality on missing property
# Inspired by JanusGraph #2205
# =============================================================================


class TestInequalityMissingProperty:
    """<> must not match nodes where property is missing (NULL)."""

    def test_neq_excludes_missing(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})")
        db.execute("INSERT (:Person {name: 'Gus'})")
        db.execute("INSERT (:Person {age: 30})")  # no name
        result = list(
            db.execute("MATCH (n:Person) WHERE n.name <> 'Alix' RETURN n.name ORDER BY n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Gus"


# =============================================================================
# Chained OR correctness
# Inspired by JanusGraph #2231
# =============================================================================


class TestChainedOr:
    """(A OR B) AND (C OR D) must not flatten to (A OR B OR C OR D)."""

    def test_chained_or_and(self, db):
        db.execute("INSERT (:T {a: true,  b: true,  c: true,  d: true})")
        db.execute("INSERT (:T {a: true,  b: false, c: true,  d: false})")
        db.execute("INSERT (:T {a: false, b: true,  c: false, d: true})")
        db.execute("INSERT (:T {a: false, b: false, c: true,  d: false})")
        result = list(
            db.execute(
                "MATCH (n:T) "
                "WHERE (n.a = true OR n.b = true) AND (n.c = false OR n.d = true) "
                "RETURN count(*) AS cnt"
            )
        )
        assert result[0]["cnt"] == 2


# =============================================================================
# Property key prefix overlap
# Inspired by JanusGraph #4401
# =============================================================================


class TestPropertyKeyPrefix:
    """Property keys that share prefixes must not interfere."""

    def test_shared_prefix_keys(self, db):
        db.execute("INSERT (:N {hello: 'world', hel: 'wor'})")
        r1 = list(db.execute("MATCH (n:N) RETURN n.hel"))
        r2 = list(db.execute("MATCH (n:N) RETURN n.hello"))
        assert r1[0]["n.hel"] == "wor"
        assert r2[0]["n.hello"] == "world"


# =============================================================================
# Property type overwrite
# Inspired by JanusGraph #4141
# =============================================================================


class TestPropertyTypeOverwrite:
    """Overwriting a property with a different type must work."""

    def test_bool_to_string(self, db):
        db.execute("INSERT (:Item {flag: true})")
        db.execute("MATCH (n:Item) SET n.flag = 'yes'")
        result = list(db.execute("MATCH (n:Item) RETURN n.flag"))
        assert result[0]["n.flag"] == "yes"

    def test_int_to_string(self, db):
        db.execute("INSERT (:Item {val: 42})")
        db.execute("MATCH (n:Item) SET n.val = 'forty-two'")
        result = list(db.execute("MATCH (n:Item) RETURN n.val"))
        assert result[0]["n.val"] == "forty-two"

    def test_string_to_int(self, db):
        db.execute("INSERT (:Item {val: 'hello'})")
        db.execute("MATCH (n:Item) SET n.val = 99")
        result = list(db.execute("MATCH (n:Item) RETURN n.val"))
        assert result[0]["n.val"] == 99


# =============================================================================
# Escaped quotes in property values
# Inspired by Dgraph #9405
# =============================================================================


class TestEscapedQuotes:
    """Quoted characters in property values must roundtrip correctly."""

    def test_double_quotes_in_single_quoted(self, db):
        db.execute("""INSERT (:Book {title: 'The "Problem" of Knowledge'})""")
        result = list(db.execute("MATCH (b:Book) RETURN b.title"))
        assert '"' in result[0]["b.title"]

    def test_single_quotes_in_double_quoted(self, db):
        db.execute("""INSERT (:Book {title: "It's a Test"})""")
        result = list(db.execute("MATCH (b:Book) RETURN b.title"))
        assert "'" in result[0]["b.title"]


# =============================================================================
# SKIP and SKIP + LIMIT
# =============================================================================


class TestSkipLimit:
    """Pagination with SKIP and LIMIT."""

    def test_skip(self, db):
        for i in range(5):
            db.execute(f"INSERT (:Item {{seq: {i}}})")
        result = list(db.execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq SKIP 2"))
        assert len(result) == 3
        assert result[0]["n.seq"] == 2

    def test_skip_plus_limit(self, db):
        for i in range(10):
            db.execute(f"INSERT (:Item {{seq: {i}}})")
        result = list(db.execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq SKIP 3 LIMIT 2"))
        assert len(result) == 2
        assert result[0]["n.seq"] == 3
        assert result[1]["n.seq"] == 4


# =============================================================================
# RETURN DISTINCT
# =============================================================================


class TestReturnDistinct:
    """DISTINCT removes exact duplicates."""

    def test_distinct_values(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 2})")
        result = list(db.execute("MATCH (n:N) RETURN DISTINCT n.val ORDER BY n.val"))
        assert len(result) == 2

    def test_distinct_collapses_nulls(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N)")
        db.execute("INSERT (:N)")
        result = list(db.execute("MATCH (n:N) RETURN DISTINCT n.val"))
        assert len(result) == 2


# =============================================================================
# WITH clause
# =============================================================================


class TestWithClause:
    """WITH projects and filters between query parts."""

    def test_with_renames(self, db):
        db.execute("INSERT (:Person {name: 'Alix', age: 30})")
        result = list(
            db.execute(
                "MATCH (p:Person) "
                "WITH p.name AS person_name, p.age AS person_age "
                "RETURN person_name, person_age"
            )
        )
        assert result[0]["person_name"] == "Alix"
        assert result[0]["person_age"] == 30

    def test_with_filters(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 2})")
        db.execute("INSERT (:N {val: 3})")
        result = list(db.execute("MATCH (n:N) WITH n WHERE n.val > 1 RETURN n.val ORDER BY n.val"))
        assert len(result) == 2
        assert result[0]["n.val"] == 2


# =============================================================================
# ORDER BY with mixed types
# Inspired by Memgraph #3888
# =============================================================================


class TestOrderByMixedTypes:
    """ORDER BY with heterogeneous types must not crash."""

    def test_mixed_types_no_crash(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 'hello'})")
        db.execute("INSERT (:N {val: true})")
        result = list(db.execute("MATCH (n:N) RETURN n.val ORDER BY n.val"))
        assert len(result) == 3


# =============================================================================
# User workflow: multi-clause MATCH + CREATE relationship
# From Grafeo user josema-xyz
# =============================================================================


class TestMultiMatchCreateEdge:
    """MATCH(a), (b) CREATE (a)-[:R]->(b) must not create phantom nodes."""

    def test_match_match_create_edge(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})")
        db.execute("INSERT (:Person {name: 'Gus'})")
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)"
        )
        nodes = list(db.execute("MATCH (n:Person) RETURN count(n) AS cnt"))
        edges = list(db.execute("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt"))
        assert nodes[0]["cnt"] == 2, "Should still have 2 nodes"
        assert edges[0]["cnt"] == 1, "Should have 1 edge"

    def test_match_create_multiple_edges(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})")
        db.execute("INSERT (:Person {name: 'Gus'})")
        db.execute("INSERT (:Person {name: 'Vincent'})")
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person) "
            "WHERE b.name <> 'Alix' "
            "INSERT (a)-[:KNOWS]->(b)"
        )
        edges = list(
            db.execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() RETURN count(r) AS cnt")
        )
        assert edges[0]["cnt"] == 2


# =============================================================================
# User workflow: negative numeric literals
# From Grafeo user janit: geographic coordinates
# =============================================================================


class TestNegativeNumerics:
    """Negative integers and floats must be preserved."""

    def test_negative_integer(self, db):
        db.execute("INSERT (:Location {lat: -33, lon: 151})")
        result = list(db.execute("MATCH (l:Location) RETURN l.lat, l.lon"))
        assert result[0]["l.lat"] == -33
        assert result[0]["l.lon"] == 151

    def test_negative_float(self, db):
        db.execute("INSERT (:Location {lat: -33.8688, lon: 151.2093})")
        result = list(db.execute("MATCH (l:Location) RETURN l.lat"))
        assert result[0]["l.lat"] < 0, "Negative latitude must be preserved"

    def test_filter_on_negative(self, db):
        db.execute("INSERT (:Loc {lat: -33.0})")
        db.execute("INSERT (:Loc {lat: 48.8})")
        result = list(db.execute("MATCH (l:Loc) WHERE l.lat < 0 RETURN l.lat"))
        assert len(result) == 1

    def test_merge_with_negative(self, db):
        db.execute("MERGE (:Temp {val: -42})")
        db.execute("MERGE (:Temp {val: -42})")
        result = list(db.execute("MATCH (n:Temp) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1


# =============================================================================
# User workflow: count(*) vs count(n) agreement
# From Grafeo user josema-xyz
# =============================================================================


class TestCountVariants:
    """count(*) and count(n) must agree."""

    def test_count_star(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 2})")
        result = list(db.execute("MATCH (n:N) RETURN count(*) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_count_variable(self, db):
        db.execute("INSERT (:N {val: 1})")
        db.execute("INSERT (:N {val: 2})")
        result = list(db.execute("MATCH (n:N) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_count_star_equals_count_variable(self, db):
        for i in range(5):
            db.execute(f"INSERT (:N {{val: {i}}})")
        r1 = list(db.execute("MATCH (n:N) RETURN count(*) AS cnt"))
        r2 = list(db.execute("MATCH (n:N) RETURN count(n) AS cnt"))
        assert r1[0]["cnt"] == r2[0]["cnt"], "count(*) and count(n) must agree"


# =============================================================================
# User workflow: batch upsert via UNWIND + MERGE + SET
# From Grafeo user Imaclean74: code dependency graph bulk import
# =============================================================================


class TestBatchUpsert:
    """UNWIND + MERGE + SET for bulk import workflows."""

    def test_unwind_merge_set(self, db):
        db.execute("UNWIND [1, 2, 3] AS i MERGE (n:Item {key: i}) SET n.updated = true")
        result = list(db.execute("MATCH (n:Item) WHERE n.updated = true RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_unwind_merge_second_pass_updates(self, db):
        db.execute("UNWIND [1, 2, 3] AS i MERGE (n:Item {key: i}) SET n.ver = 1")
        db.execute("UNWIND [1, 2, 3] AS i MERGE (n:Item {key: i}) SET n.ver = 2")
        nodes = list(db.execute("MATCH (n:Item) RETURN count(n) AS cnt"))
        assert nodes[0]["cnt"] == 3, "No duplicates from second MERGE"
        updated = list(db.execute("MATCH (n:Item) WHERE n.ver = 2 RETURN count(n) AS cnt"))
        assert updated[0]["cnt"] == 3, "All should have ver=2"


# =============================================================================
# User workflow: labels() and type() as grouping keys
# From Grafeo user Imaclean74: schema introspection
# =============================================================================


class TestAggregationWithFunctions:
    """Grouping by labels(n) and type(r) with count()."""

    def test_group_by_labels_count(self, db):
        db.execute("INSERT (:Person {name: 'Alix'})")
        db.execute("INSERT (:Person {name: 'Gus'})")
        db.execute("INSERT (:City {name: 'Amsterdam'})")
        result = list(
            db.execute("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label")
        )
        assert len(result) == 2
        assert result[0]["label"] == "City"
        assert result[0]["cnt"] == 1
        assert result[1]["label"] == "Person"
        assert result[1]["cnt"] == 2

    def test_group_by_type_count(self, db):
        db.execute("INSERT (:A)-[:FOLLOWS]->(:B)")
        db.execute("INSERT (:C)-[:FOLLOWS]->(:D)")
        db.execute("INSERT (:E)-[:BLOCKS]->(:F)")
        result = list(
            db.execute("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY t")
        )
        assert len(result) == 2
