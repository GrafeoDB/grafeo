"""Regression tests for GraphQL query support, inspired by bugs in other databases.

Covers aggregation, pagination, mutation correctness, type handling,
and filter translation edge cases.

Run: pytest tests/lpg/graphql/test_regression_external.py -v
"""


# =============================================================================
# Aggregation count with filters
# Inspired by neo4j/graphql #6917: count disagrees with filtered results.
# =============================================================================


class TestAggregationWithFilters:
    """Aggregate counts must agree with filtered result sets."""

    def test_count_matches_filtered_edges(self, db):
        """count() of filtered edges must match actual result rows."""
        alix = db.create_node(["Person"], {"name": "Alix"})
        car1 = db.create_node(["Car"], {"name": "Tesla", "color": "red"})
        car2 = db.create_node(["Car"], {"name": "BMW", "color": "blue"})
        db.create_edge(alix.id, car1.id, "OWNS", {})
        db.create_edge(alix.id, car2.id, "OWNS", {})

        # Count matching red cars
        result = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'})-[:OWNS]->(c:Car) "
                "WHERE c.color = 'red' "
                "RETURN count(c) AS cnt"
            )
        )
        assert result[0]["cnt"] == 1

    def test_count_zero_when_no_match(self, db):
        """When filter excludes all edges, count must be 0."""
        alix = db.create_node(["Person"], {"name": "Alix"})
        car = db.create_node(["Car"], {"name": "BMW", "color": "blue"})
        db.create_edge(alix.id, car.id, "OWNS", {})

        result = list(
            db.execute(
                "MATCH (p:Person {name: 'Alix'})-[:OWNS]->(c:Car) "
                "WHERE c.color = 'red' "
                "RETURN count(c) AS cnt"
            )
        )
        assert result[0]["cnt"] == 0


# =============================================================================
# Edge property updates
# Inspired by neo4j/graphql #6981: relationship property updates silently fail.
# =============================================================================


class TestEdgePropertyUpdate:
    """Edge properties must be updatable."""

    def test_set_edge_property(self, db):
        a = db.create_node(["A"], {"name": "a"})
        b = db.create_node(["B"], {"name": "b"})
        db.create_edge(a.id, b.id, "REL", {"order": 1})

        db.execute("MATCH (:A {name: 'a'})-[r:REL]->(:B {name: 'b'}) SET r.order = 2")
        result = list(db.execute("MATCH (:A {name: 'a'})-[r:REL]->(:B {name: 'b'}) RETURN r.order"))
        assert result[0]["r.order"] == 2


# =============================================================================
# LIMIT + ORDER BY
# Inspired by dgraph #9239: first/limit silently capped when sorting.
# =============================================================================


class TestLimitWithOrderBy:
    """LIMIT must be respected with ORDER BY."""

    def test_limit_with_order(self, db):
        for i in range(20):
            db.create_node(["Item"], {"seq": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 5"))
        assert len(result) == 5
        assert result[0]["n.seq"] == 0
        assert result[4]["n.seq"] == 4

    def test_limit_without_order(self, db):
        for i in range(20):
            db.create_node(["Item"], {"seq": i})
        result = list(db.execute("MATCH (n:Item) RETURN n.seq LIMIT 5"))
        assert len(result) == 5


# =============================================================================
# Idempotent property SET
# Inspired by dgraph #9519: re-setting same value breaks mutation.
# =============================================================================


class TestIdempotentSet:
    """SET to same value must not break other updates."""

    def test_set_same_value_with_other_change(self, db):
        db.execute("INSERT (:N {code: 'X', note: 'initial'})")
        db.execute("MATCH (n:N {code: 'X'}) SET n.code = 'X', n.note = 'updated'")
        result = list(db.execute("MATCH (n:N {code: 'X'}) RETURN n.note"))
        assert result[0]["n.note"] == "updated"


# =============================================================================
# DETACH DELETE with multiple relationships
# Inspired by dgraph #9552: delete panics on nodes with inverse edges.
# =============================================================================


class TestDetachDeleteMultipleRelationships:
    """DETACH DELETE on a node with many edge types must not crash."""

    def test_detach_delete_multi_edge_node(self, db):
        center = db.create_node(["Hub"], {"name": "center"})
        for i in range(5):
            spoke = db.create_node(["Spoke"], {"idx": i})
            db.create_edge(center.id, spoke.id, "CONNECTS", {})
            db.create_edge(spoke.id, center.id, "BACK", {})

        db.execute("MATCH (h:Hub {name: 'center'}) DETACH DELETE h")
        result = list(db.execute("MATCH (h:Hub) RETURN count(h) AS cnt"))
        assert result[0]["cnt"] == 0
        edges = list(db.execute("MATCH ()-[r:CONNECTS]->() RETURN count(r) AS cnt"))
        assert edges[0]["cnt"] == 0


# =============================================================================
# Integer type preservation
# Inspired by neo4j/graphql #6615: Int returned as non-integer.
# =============================================================================


class TestIntegerTypePreservation:
    """Integer properties must retain their type through queries."""

    def test_integer_roundtrip(self, db):
        db.execute("INSERT (:Product {order: 3})")
        result = list(db.execute("MATCH (p:Product) RETURN p.order"))
        assert isinstance(result[0]["p.order"], int)
        assert result[0]["p.order"] == 3

    def test_float_not_confused_with_int(self, db):
        db.execute("INSERT (:Product {price: 3.14})")
        result = list(db.execute("MATCH (p:Product) RETURN p.price"))
        assert isinstance(result[0]["p.price"], float)


# =============================================================================
# Filter for non-existence of relationships
# Inspired by neo4j/graphql #7120.
# =============================================================================


class TestNoRelationshipFilter:
    """Nodes with zero relationships of a type must be filterable."""

    def test_not_exists_pattern(self, db):
        db.execute("INSERT (:Movie {title: 'Solo'})")
        db.execute("INSERT (:Movie {title: 'Ensemble'})<-[:ACTED_IN]-(:Actor {name: 'Alix'})")
        result = list(
            db.execute(
                "MATCH (m:Movie) "
                "WHERE NOT EXISTS { MATCH (m)<-[:ACTED_IN]-(:Actor) } "
                "RETURN m.title"
            )
        )
        # Only 'Solo' has no actors
        titles = [r["m.title"] for r in result]
        assert "Solo" in titles
        assert "Ensemble" not in titles


# =============================================================================
# Multi-hop path filter
# Inspired by neo4j/graphql #7045: chained resolver variables undefined.
# =============================================================================


class TestMultiHopFilter:
    """Filtering through multi-hop relationships must work."""

    def test_two_hop_filter(self, db):
        db.execute("INSERT (:A {name: 'a1'})-[:HAS]->(:B {name: 'b1'})-[:HAS]->(:C {name: 'test'})")
        db.execute(
            "INSERT (:A {name: 'a2'})-[:HAS]->(:B {name: 'b2'})-[:HAS]->(:C {name: 'other'})"
        )
        result = list(
            db.execute("MATCH (a:A)-[:HAS]->(:B)-[:HAS]->(c:C) WHERE c.name = 'test' RETURN a.name")
        )
        assert len(result) == 1
        assert result[0]["a.name"] == "a1"


# =============================================================================
# Bidirectional edge update consistency
# Inspired by dgraph #9338: inverse relationship half-updated.
# =============================================================================


class TestBidirectionalEdgeUpdate:
    """Replacing an edge must update both directions."""

    def test_replace_forward_edge(self, db):
        db.execute("INSERT (:A {name: 'a'})-[:LINKED]->(:B {name: 'b1'})")
        db.execute("INSERT (:B {name: 'b2'})")

        # Delete old, create new
        db.execute("MATCH (:A {name: 'a'})-[r:LINKED]->() DELETE r")
        db.execute("MATCH (a:A {name: 'a'}), (b:B {name: 'b2'}) INSERT (a)-[:LINKED]->(b)")

        # Forward: a -> b2
        fwd = list(db.execute("MATCH (:A {name: 'a'})-[:LINKED]->(b) RETURN b.name"))
        assert len(fwd) == 1
        assert fwd[0]["b.name"] == "b2"

        # Backward: b1 should have no incoming LINKED
        bwd = list(db.execute("MATCH (:B {name: 'b1'})<-[:LINKED]-(a) RETURN a.name"))
        assert len(bwd) == 0
