"""Cypher implementation of query tests.

Tests pattern matching, paths, and aggregations using Cypher query language.
"""

import pytest
from tests.python.bases.test_queries import BaseQueriesTest


class TestCypherQueries(BaseQueriesTest):
    """Cypher implementation of query tests."""

    def execute_query(self, db, query):
        """Execute query using Cypher parser."""
        return db.execute_cypher(query)

    # =========================================================================
    # SETUP METHODS
    # =========================================================================

    def setup_pattern_graph(self, db):
        """Set up test data for pattern tests."""
        alice = db.create_node(["Person"], {"name": "Alice", "age": 30, "city": "NYC"})
        bob = db.create_node(["Person"], {"name": "Bob", "age": 25, "city": "LA"})
        charlie = db.create_node(["Person"], {"name": "Charlie", "age": 35, "city": "NYC"})

        acme = db.create_node(["Company"], {"name": "Acme Corp", "founded": 2010})
        globex = db.create_node(["Company"], {"name": "Globex Inc", "founded": 2015})

        db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})
        db.create_edge(bob.id, charlie.id, "KNOWS", {"since": 2021})
        db.create_edge(alice.id, charlie.id, "KNOWS", {"since": 2019})

        db.create_edge(alice.id, acme.id, "WORKS_AT", {"role": "Engineer"})
        db.create_edge(bob.id, globex.id, "WORKS_AT", {"role": "Manager"})
        db.create_edge(charlie.id, acme.id, "WORKS_AT", {"role": "Director"})

        return {"alice": alice, "bob": bob, "charlie": charlie, "acme": acme, "globex": globex}

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

        db.create_edge(a.id, d.id, "DIRECT", {})
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
        return f"MATCH (n:{label}) RETURN n.{return_prop}"

    def match_where_query(self, label: str, prop: str, op: str, value, return_prop: str = "name") -> str:
        value_str = f"'{value}'" if isinstance(value, str) else str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {value_str} RETURN n.{return_prop}"

    def match_and_query(self, label: str, prop1: str, op1: str, value1, prop2: str, op2: str, value2, return_prop: str = "name") -> str:
        val1 = f"'{value1}'" if isinstance(value1, str) else value1
        val2 = f"'{value2}'" if isinstance(value2, str) else value2
        return f"MATCH (p:{label}) WHERE p.{prop1} {op1} {val1} AND p.{prop2} {op2} {val2} RETURN p.{return_prop}"

    def match_relationship_query(self, from_label: str, rel_type: str, to_label: str, return_from: str = "name", return_to: str = "name") -> str:
        return f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) RETURN a.{return_from} AS from_{return_from}, b.{return_to} AS to_{return_to}"

    def match_relationship_with_props_query(self, from_label: str, rel_type: str, to_label: str, rel_prop: str, op: str, value) -> str:
        val = f"'{value}'" if isinstance(value, str) else value
        return f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) WHERE r.{rel_prop} {op} {val} RETURN a.name, b.name, r.{rel_prop}"

    def match_multi_hop_query(self, start_label: str, rel_type: str, end_label: str) -> str:
        return f"MATCH (a:{start_label})-[:{rel_type}]->(b:{start_label})-[:{rel_type}]->(c:{end_label}) RETURN a.name, b.name, c.name"

    # =========================================================================
    # PATH QUERIES
    # =========================================================================

    def variable_length_path_query(self, start_label: str, start_prop: str, start_value, rel_type: str, end_label: str, min_hops: int, max_hops: int) -> str:
        val = f"'{start_value}'" if isinstance(start_value, str) else start_value
        return f"MATCH (start:{start_label} {{{start_prop}: {val}}})-[:{rel_type}*{min_hops}..{max_hops}]->(end:{end_label}) RETURN end.name"

    def shortest_path_query(self, start_label: str, start_prop: str, start_value, end_label: str, end_prop: str, end_value) -> str:
        start_val = f"'{start_value}'" if isinstance(start_value, str) else start_value
        end_val = f"'{end_value}'" if isinstance(end_value, str) else end_value
        return f"MATCH p = shortestPath((a:{start_label} {{{start_prop}: {start_val}}})-[*]-(d:{end_label} {{{end_prop}: {end_val}}})) RETURN length(p) AS path_length"

    # =========================================================================
    # AGGREGATION QUERIES
    # =========================================================================

    def count_query(self, label: str) -> str:
        return f"MATCH (n:{label}) RETURN count(n) AS cnt"

    def count_distinct_query(self, label: str, prop: str) -> str:
        return f"MATCH (p:{label}) RETURN count(DISTINCT p.{prop}) AS cities"

    def sum_avg_query(self, label: str, prop: str) -> str:
        return f"MATCH (p:{label}) RETURN sum(p.{prop}) AS total, avg(p.{prop}) AS average"

    def min_max_query(self, label: str, prop: str) -> str:
        return f"MATCH (p:{label}) RETURN min(p.{prop}) AS minimum, max(p.{prop}) AS maximum"

    def group_by_query(self, label: str, group_prop: str) -> str:
        return f"MATCH (p:{label}) RETURN p.{group_prop}, count(p) AS cnt ORDER BY cnt DESC"


# =============================================================================
# CYPHER-SPECIFIC TESTS
# =============================================================================

class TestCypherSpecificPatterns:
    """Cypher-specific pattern tests."""

    def test_cypher_inline_properties(self, pattern_db):
        """Test Cypher inline property matching."""
        result = pattern_db.execute_cypher("MATCH (p:Person {city: 'NYC'}) RETURN p.name")
        rows = list(result)
        names = [r["p.name"] for r in rows]
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names

    def test_cypher_with_clause(self, pattern_db):
        """Test Cypher WITH clause for query chaining."""
        result = pattern_db.execute_cypher(
            "MATCH (p:Person) WITH p.name AS name, p.age AS age WHERE age > 25 RETURN name, age ORDER BY age"
        )
        rows = list(result)
        assert len(rows) == 2

    def test_cypher_unwind(self, db):
        """Test Cypher UNWIND list."""
        result = db.execute_cypher("UNWIND [1, 2, 3] AS x RETURN x")
        rows = list(result)
        assert len(rows) == 3

    def test_cypher_optional_match(self, pattern_db):
        """Test Cypher OPTIONAL MATCH."""
        pattern_db.create_node(["Person"], {"name": "Diana", "age": 40})
        result = pattern_db.execute_cypher(
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name"
        )
        rows = list(result)
        assert len(rows) == 4


class TestCypherSpecificAggregations:
    """Cypher-specific aggregation tests."""

    def test_cypher_collect(self, db):
        """Test Cypher collect() aggregation."""
        db.create_node(["Person"], {"name": "Alice"})
        db.create_node(["Person"], {"name": "Bob"})
        db.create_node(["Person"], {"name": "Charlie"})

        result = db.execute_cypher("MATCH (p:Person) RETURN collect(p.name) AS names")
        rows = list(result)
        names = rows[0]["names"]
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    def test_cypher_collect_distinct(self, db):
        """Test Cypher collect(DISTINCT) aggregation."""
        db.create_node(["Person"], {"name": "Alice", "city": "NYC"})
        db.create_node(["Person"], {"name": "Bob", "city": "NYC"})
        db.create_node(["Person"], {"name": "Charlie", "city": "LA"})

        result = db.execute_cypher("MATCH (p:Person) RETURN collect(DISTINCT p.city) AS cities")
        rows = list(result)
        cities = rows[0]["cities"]
        assert len(cities) == 2

    def test_cypher_percentile(self, db):
        """Test Cypher percentile functions."""
        for age in [20, 25, 30, 35, 40, 45, 50]:
            db.create_node(["Person"], {"name": f"P{age}", "age": age})

        result = db.execute_cypher(
            "MATCH (p:Person) RETURN percentileDisc(p.age, 0.5) AS median_disc, percentileCont(p.age, 0.5) AS median_cont"
        )
        rows = list(result)
        assert 30 <= rows[0]["median_disc"] <= 40

    def test_cypher_stdev(self, db):
        """Test Cypher standard deviation."""
        db.create_node(["Person"], {"name": "A", "score": 10})
        db.create_node(["Person"], {"name": "B", "score": 20})
        db.create_node(["Person"], {"name": "C", "score": 30})

        result = db.execute_cypher("MATCH (p:Person) RETURN stdev(p.score) AS sd")
        rows = list(result)
        assert 8 <= rows[0]["sd"] <= 12

    def test_cypher_head_tail(self, db):
        """Test Cypher head() and tail() on lists."""
        result = db.execute_cypher("WITH [1, 2, 3, 4, 5] AS nums RETURN head(nums) AS first, tail(nums) AS rest")
        rows = list(result)
        assert rows[0]["first"] == 1
        assert rows[0]["rest"] == [2, 3, 4, 5]
