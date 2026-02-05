"""Cypher implementation of mutation tests.

Tests CRUD operations using Cypher query language.
"""

import pytest
from tests.python.bases.test_mutations import BaseMutationsTest


class TestCypherMutations(BaseMutationsTest):
    """Cypher implementation of mutation tests."""

    def execute_query(self, db, query):
        """Execute query using Cypher parser."""
        return db.execute_cypher(query)

    def create_node_query(self, labels: list[str], props: dict) -> str:
        """Cypher: CREATE (:<labels> {<props>}) RETURN n"""
        label_str = ":".join(labels) if labels else ""
        if label_str:
            label_str = f":{label_str}"

        prop_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                prop_parts.append(f"{k}: '{v}'")
            elif isinstance(v, bool):
                prop_parts.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                prop_parts.append(f"{k}: null")
            else:
                prop_parts.append(f"{k}: {v}")

        prop_str = ", ".join(prop_parts)
        return f"CREATE (n{label_str} {{{prop_str}}}) RETURN n"

    def match_node_query(self, label: str, return_prop: str = "name") -> str:
        return f"MATCH (n:{label}) RETURN n.{return_prop}"

    def match_where_query(self, label: str, prop: str, op: str, value, return_prop: str = "name") -> str:
        value_str = f"'{value}'" if isinstance(value, str) else str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {value_str} RETURN n.{return_prop}"

    def delete_node_query(self, label: str, prop: str, value) -> str:
        value_str = f"'{value}'" if isinstance(value, str) else str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} = {value_str} DELETE n"

    def create_edge_query(self, from_label: str, from_prop: str, from_value, to_label: str, to_prop: str, to_value, edge_type: str, edge_props: dict) -> str:
        from_val = f"'{from_value}'" if isinstance(from_value, str) else from_value
        to_val = f"'{to_value}'" if isinstance(to_value, str) else to_value

        prop_parts = []
        for k, v in edge_props.items():
            if isinstance(v, str):
                prop_parts.append(f"{k}: '{v}'")
            else:
                prop_parts.append(f"{k}: {v}")
        prop_str = ", ".join(prop_parts) if prop_parts else ""

        if prop_str:
            return f"MATCH (a:{from_label}), (b:{to_label}) WHERE a.{from_prop} = {from_val} AND b.{to_prop} = {to_val} CREATE (a)-[r:{edge_type} {{{prop_str}}}]->(b) RETURN r"
        else:
            return f"MATCH (a:{from_label}), (b:{to_label}) WHERE a.{from_prop} = {from_val} AND b.{to_prop} = {to_val} CREATE (a)-[r:{edge_type}]->(b) RETURN r"

    def update_node_query(self, label: str, match_prop: str, match_value, set_prop: str, set_value) -> str:
        match_val = f"'{match_value}'" if isinstance(match_value, str) else match_value
        set_val = f"'{set_value}'" if isinstance(set_value, str) else set_value
        return f"MATCH (n:{label}) WHERE n.{match_prop} = {match_val} SET n.{set_prop} = {set_val} RETURN n"


# =============================================================================
# CYPHER-SPECIFIC MUTATION TESTS
# =============================================================================

class TestCypherSpecificMutations:
    """Cypher-specific mutation tests."""

    def test_cypher_create_syntax(self, db):
        """Test Cypher CREATE syntax."""
        result = db.execute_cypher("CREATE (n:Person {name: 'CreateTest', age: 42}) RETURN n")
        rows = list(result)
        assert len(rows) == 1

        result = db.execute_cypher("MATCH (n:Person) WHERE n.name = 'CreateTest' RETURN n.age")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.age"] == 42

    def test_cypher_merge(self, db):
        """Test Cypher MERGE (create if not exists)."""
        db.execute_cypher("MERGE (c:City {name: 'NYC'}) RETURN c")
        db.execute_cypher("MERGE (c:City {name: 'NYC'}) RETURN c")

        result = db.execute_cypher("MATCH (c:City) RETURN count(c) AS cnt")
        rows = list(result)
        assert rows[0]["cnt"] == 1

    def test_cypher_set(self, db):
        """Test Cypher SET for property update."""
        db.execute_cypher("CREATE (p:Person {name: 'SetTest', verified: false})")
        db.execute_cypher("MATCH (p:Person {name: 'SetTest'}) SET p.verified = true")

        result = db.execute_cypher("MATCH (p:Person {name: 'SetTest'}) RETURN p.verified")
        rows = list(result)
        assert rows[0]["p.verified"] is True

    def test_cypher_set_add_property(self, db):
        """Test Cypher SET to add new property."""
        db.execute_cypher("CREATE (p:Person {name: 'AddProp'})")
        db.execute_cypher("MATCH (p:Person {name: 'AddProp'}) SET p.newProp = 'added'")

        result = db.execute_cypher("MATCH (p:Person {name: 'AddProp'}) RETURN p.newProp")
        rows = list(result)
        assert rows[0]["p.newProp"] == "added"

    def test_cypher_remove(self, db):
        """Test Cypher REMOVE property."""
        db.execute_cypher("CREATE (p:Person {name: 'RemoveTest', toRemove: 'value'})")
        db.execute_cypher("MATCH (p:Person {name: 'RemoveTest'}) REMOVE p.toRemove")

        result = db.execute_cypher("MATCH (p:Person {name: 'RemoveTest'}) RETURN p.toRemove")
        rows = list(result)
        assert rows[0].get("p.toRemove") is None

    def test_cypher_detach_delete(self, db):
        """Test Cypher DETACH DELETE (deletes node and all relationships)."""
        db.execute_cypher("CREATE (a:Node {name: 'A'})")
        db.execute_cypher("CREATE (b:Node {name: 'B'})")
        db.execute_cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:CONNECTED]->(b)")
        db.execute_cypher("MATCH (n:Node {name: 'A'}) DETACH DELETE n")

        result = db.execute_cypher("MATCH (n:Node {name: 'A'}) RETURN n")
        rows = list(result)
        assert len(rows) == 0

        result = db.execute_cypher("MATCH (n:Node {name: 'B'}) RETURN n")
        rows = list(result)
        assert len(rows) == 1
