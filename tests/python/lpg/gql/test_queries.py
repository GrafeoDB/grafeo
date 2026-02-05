"""GQL implementation of query tests.

Tests pattern matching, paths, and aggregations using GQL (ISO standard) query language.
"""

import pytest
from tests.python.bases.test_queries import BaseQueriesTest


class TestGQLQueries(BaseQueriesTest):
    """GQL implementation of query tests."""

    # =========================================================================
    # SETUP METHODS
    # =========================================================================

    def setup_pattern_graph(self, db):
        """Set up test data for pattern tests."""
        # Create Person nodes
        alice = db.create_node(["Person"], {
            "name": "Alice", "age": 30, "city": "NYC"
        })
        bob = db.create_node(["Person"], {
            "name": "Bob", "age": 25, "city": "LA"
        })
        charlie = db.create_node(["Person"], {
            "name": "Charlie", "age": 35, "city": "NYC"
        })

        # Create Company nodes
        acme = db.create_node(["Company"], {
            "name": "Acme Corp", "founded": 2010
        })
        globex = db.create_node(["Company"], {
            "name": "Globex Inc", "founded": 2015
        })

        # Create KNOWS edges
        db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})
        db.create_edge(bob.id, charlie.id, "KNOWS", {"since": 2021})
        db.create_edge(alice.id, charlie.id, "KNOWS", {"since": 2019})

        # Create WORKS_AT edges
        db.create_edge(alice.id, acme.id, "WORKS_AT", {"role": "Engineer"})
        db.create_edge(bob.id, globex.id, "WORKS_AT", {"role": "Manager"})
        db.create_edge(charlie.id, acme.id, "WORKS_AT", {"role": "Director"})

        return {
            "alice": alice, "bob": bob, "charlie": charlie,
            "acme": acme, "globex": globex,
        }

    def setup_chain_graph(self, db):
        """Set up a chain graph: a -> b -> c -> d."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})

        db.create_edge(a.id, b.id, "NEXT", {})
        db.create_edge(b.id, c.id, "NEXT", {})
        db.create_edge(c.id, d.id, "NEXT", {})

        return {"a": a.id, "b": b.id, "c": c.id, "d": d.id}

    def setup_multi_path_graph(self, db):
        """Set up a graph with multiple paths."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})

        # Direct path
        db.create_edge(a.id, d.id, "DIRECT", {})

        # Longer path
        db.create_edge(a.id, b.id, "STEP", {})
        db.create_edge(b.id, c.id, "STEP", {})
        db.create_edge(c.id, d.id, "STEP", {})

        return {"a": a.id, "b": b.id, "c": c.id, "d": d.id}

    def setup_aggregation_data(self, db):
        """Set up test data for aggregation tests."""
        db.create_node(["Person"], {"name": "Alice", "age": 30, "city": "NYC"})
        db.create_node(["Person"], {"name": "Bob", "age": 25, "city": "LA"})
        db.create_node(["Person"], {"name": "Charlie", "age": 35, "city": "NYC"})

    # =========================================================================
    # PATTERN QUERIES
    # =========================================================================

    def match_label_query(self, label: str, return_prop: str = "name") -> str:
        """GQL: MATCH (n:<label>) RETURN n.<prop>"""
        return f"MATCH (n:{label}) RETURN n.{return_prop}"

    def match_where_query(
        self, label: str, prop: str, op: str, value, return_prop: str = "name"
    ) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> <op> <value> RETURN n.<prop>"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {value_str} RETURN n.{return_prop}"

    def match_and_query(
        self,
        label: str,
        prop1: str,
        op1: str,
        value1,
        prop2: str,
        op2: str,
        value2,
        return_prop: str = "name",
    ) -> str:
        """GQL: MATCH with AND in WHERE clause."""
        val1 = f"'{value1}'" if isinstance(value1, str) else value1
        val2 = f"'{value2}'" if isinstance(value2, str) else value2
        return (
            f"MATCH (p:{label}) "
            f"WHERE p.{prop1} {op1} {val1} AND p.{prop2} {op2} {val2} "
            f"RETURN p.{return_prop}"
        )

    def match_relationship_query(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        return_from: str = "name",
        return_to: str = "name",
    ) -> str:
        """GQL: MATCH (a)-[r:<rel_type>]->(b) RETURN ..."""
        return (
            f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
            f"RETURN a.{return_from} AS from_{return_from}, b.{return_to} AS to_{return_to}"
        )

    def match_relationship_with_props_query(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        rel_prop: str,
        op: str,
        value,
    ) -> str:
        """GQL: MATCH with relationship property filter."""
        val = f"'{value}'" if isinstance(value, str) else value
        return (
            f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
            f"WHERE r.{rel_prop} {op} {val} "
            f"RETURN a.name, b.name, r.{rel_prop}"
        )

    def match_multi_hop_query(
        self,
        start_label: str,
        rel_type: str,
        end_label: str,
    ) -> str:
        """GQL: 2-hop path pattern."""
        return (
            f"MATCH (a:{start_label})-[:{rel_type}]->(b:{start_label})"
            f"-[:{rel_type}]->(c:{end_label}) "
            f"RETURN a.name, b.name, c.name"
        )

    # =========================================================================
    # PATH QUERIES
    # =========================================================================

    def variable_length_path_query(
        self,
        start_label: str,
        start_prop: str,
        start_value,
        rel_type: str,
        end_label: str,
        min_hops: int,
        max_hops: int,
    ) -> str:
        """GQL: Variable-length path query."""
        val = f"'{start_value}'" if isinstance(start_value, str) else start_value
        return (
            f"MATCH (start:{start_label} {{{start_prop}: {val}}})"
            f"-[:{rel_type}*{min_hops}..{max_hops}]->(end:{end_label}) "
            f"RETURN end.name"
        )

    def shortest_path_query(
        self,
        start_label: str,
        start_prop: str,
        start_value,
        end_label: str,
        end_prop: str,
        end_value,
    ) -> str:
        """GQL: Shortest path query."""
        start_val = f"'{start_value}'" if isinstance(start_value, str) else start_value
        end_val = f"'{end_value}'" if isinstance(end_value, str) else end_value
        return (
            f"MATCH p = shortestPath("
            f"(a:{start_label} {{{start_prop}: {start_val}}})"
            f"-[*]-"
            f"(d:{end_label} {{{end_prop}: {end_val}}})"
            f") RETURN length(p) AS path_length"
        )

    # =========================================================================
    # AGGREGATION QUERIES
    # =========================================================================

    def count_query(self, label: str) -> str:
        """GQL: MATCH (n:<label>) RETURN count(n) AS cnt"""
        return f"MATCH (n:{label}) RETURN count(n) AS cnt"

    def count_distinct_query(self, label: str, prop: str) -> str:
        """GQL: MATCH (p:<label>) RETURN count(DISTINCT p.<prop>) AS cities"""
        return f"MATCH (p:{label}) RETURN count(DISTINCT p.{prop}) AS cities"

    def sum_avg_query(self, label: str, prop: str) -> str:
        """GQL: RETURN sum(p.<prop>) AS total, avg(p.<prop>) AS average"""
        return (
            f"MATCH (p:{label}) "
            f"RETURN sum(p.{prop}) AS total, avg(p.{prop}) AS average"
        )

    def min_max_query(self, label: str, prop: str) -> str:
        """GQL: RETURN min(p.<prop>) AS minimum, max(p.<prop>) AS maximum"""
        return (
            f"MATCH (p:{label}) "
            f"RETURN min(p.{prop}) AS minimum, max(p.{prop}) AS maximum"
        )

    def group_by_query(self, label: str, group_prop: str) -> str:
        """GQL: RETURN p.<prop>, count(p) AS cnt ORDER BY cnt DESC"""
        return (
            f"MATCH (p:{label}) "
            f"RETURN p.{group_prop}, count(p) AS cnt "
            f"ORDER BY cnt DESC"
        )


# =============================================================================
# GQL-SPECIFIC TESTS
# =============================================================================

class TestGQLSpecificPatterns:
    """GQL-specific pattern tests."""

    def test_gql_optional_match(self, pattern_db):
        """Test GQL OPTIONAL MATCH."""
        pattern_db.create_node(["Person"], {"name": "Diana", "age": 40, "city": "Chicago"})

        result = pattern_db.execute(
            "MATCH (p:Person) "
            "OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
            "RETURN p.name, c.name"
        )
        rows = list(result)
        assert len(rows) == 4

    def test_gql_multiple_matches(self, pattern_db):
        """Test multiple MATCH clauses."""
        result = pattern_db.execute(
            "MATCH (p:Person) "
            "MATCH (c:Company) "
            "RETURN p.name, c.name"
        )
        rows = list(result)
        assert len(rows) == 6  # 3 persons x 2 companies

    def test_gql_undirected_match(self, pattern_db):
        """Test undirected relationship match."""
        result = pattern_db.execute(
            "MATCH (a:Person)-[:KNOWS]-(b:Person) "
            "RETURN a.name, b.name"
        )
        rows = list(result)
        assert len(rows) == 6  # Each edge counted twice

    def test_gql_exists_pattern(self, pattern_db):
        """Test EXISTS in WHERE clause."""
        result = pattern_db.execute(
            "MATCH (p:Person) "
            "WHERE EXISTS { MATCH (p)-[:WORKS_AT]->(:Company) } "
            "RETURN p.name"
        )
        rows = list(result)
        assert len(rows) == 3


class TestGQLSpecificPaths:
    """GQL-specific path tests."""

    def test_gql_all_shortest_paths(self, db):
        """Test GQL allShortestPaths."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})

        db.create_edge(a.id, b.id, "EDGE", {})
        db.create_edge(a.id, c.id, "EDGE", {})
        db.create_edge(b.id, d.id, "EDGE", {})
        db.create_edge(c.id, d.id, "EDGE", {})

        result = db.execute(
            "MATCH p = allShortestPaths("
            "(a:Node {name: 'a'})-[*]-(d:Node {name: 'd'})"
            ") RETURN length(p) AS len"
        )
        rows = list(result)
        assert len(rows) >= 2

    def test_gql_path_with_filter(self, db):
        """Test path query with relationship filter."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})

        db.create_edge(a.id, b.id, "GOOD", {"weight": 1})
        db.create_edge(b.id, c.id, "GOOD", {"weight": 2})
        db.create_edge(a.id, c.id, "BAD", {"weight": 10})

        result = db.execute(
            "MATCH p = (a:Node {name: 'a'})-[:GOOD*1..3]->(c:Node {name: 'c'}) "
            "RETURN length(p) AS len"
        )
        rows = list(result)
        assert len(rows) >= 1

    def test_gql_no_path_returns_empty(self, db):
        """Test that non-existent path returns no rows."""
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})

        result = db.execute(
            "MATCH (a:Node {name: 'a'})-[:EDGE]->(b:Node {name: 'b'}) "
            "RETURN a, b"
        )
        rows = list(result)
        assert len(rows) == 0


class TestGQLSpecificAggregations:
    """GQL-specific aggregation tests."""

    def test_gql_collect(self, db):
        """Test GQL collect() aggregation."""
        db.create_node(["Person"], {"name": "Alice", "age": 30})
        db.create_node(["Person"], {"name": "Bob", "age": 25})
        db.create_node(["Person"], {"name": "Charlie", "age": 35})

        result = db.execute(
            "MATCH (p:Person) RETURN collect(p.name) AS names"
        )
        rows = list(result)
        assert len(rows) == 1
        names = rows[0]["names"]
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    def test_gql_having(self, db):
        """Test GQL GROUP BY with HAVING."""
        db.create_node(["Person"], {"name": "Alice", "city": "NYC"})
        db.create_node(["Person"], {"name": "Bob", "city": "NYC"})
        db.create_node(["Person"], {"name": "Charlie", "city": "LA"})
        db.create_node(["Person"], {"name": "Diana", "city": "NYC"})

        result = db.execute(
            "MATCH (p:Person) "
            "RETURN p.city, count(p) AS cnt "
            "HAVING cnt > 1"
        )
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["p.city"] == "NYC"

    def test_gql_order_by(self, db):
        """Test GQL ORDER BY."""
        db.create_node(["Person"], {"name": "Alice", "age": 30})
        db.create_node(["Person"], {"name": "Bob", "age": 25})
        db.create_node(["Person"], {"name": "Charlie", "age": 35})

        result = db.execute(
            "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age ASC"
        )
        rows = list(result)
        assert rows[0]["p.name"] == "Bob"
        assert rows[2]["p.name"] == "Charlie"

    def test_gql_limit(self, db):
        """Test GQL LIMIT."""
        db.create_node(["Person"], {"name": "Alice"})
        db.create_node(["Person"], {"name": "Bob"})
        db.create_node(["Person"], {"name": "Charlie"})

        result = db.execute(
            "MATCH (p:Person) RETURN p.name LIMIT 2"
        )
        rows = list(result)
        assert len(rows) == 2

    def test_gql_skip(self, db):
        """Test GQL SKIP (OFFSET)."""
        db.create_node(["Person"], {"name": "Alice", "age": 30})
        db.create_node(["Person"], {"name": "Bob", "age": 25})
        db.create_node(["Person"], {"name": "Charlie", "age": 35})

        result = db.execute(
            "MATCH (p:Person) RETURN p.name ORDER BY p.age SKIP 1 LIMIT 2"
        )
        rows = list(result)
        assert len(rows) == 2
