"""GQL implementation of mutation tests.

Tests CRUD operations using GQL (ISO standard) query language.
"""

import pytest
from tests.python.bases.test_mutations import BaseMutationsTest


class TestGQLMutations(BaseMutationsTest):
    """GQL implementation of mutation tests."""

    def create_node_query(self, labels: list[str], props: dict) -> str:
        """GQL: INSERT (:<labels> {<props>})"""
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
        return f"INSERT (n{label_str} {{{prop_str}}}) RETURN n"

    def match_node_query(self, label: str, return_prop: str = "name") -> str:
        """GQL: MATCH (n:<label>) RETURN n.<prop>"""
        return f"MATCH (n:{label}) RETURN n.{return_prop}"

    def match_where_query(
        self, label: str, prop: str, op: str, value, return_prop: str = "name"
    ) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> <op> <value> RETURN n.<return_prop>"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {value_str} RETURN n.{return_prop}"

    def delete_node_query(self, label: str, prop: str, value) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> = <value> DELETE n"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} = {value_str} DELETE n"

    def create_edge_query(
        self,
        from_label: str,
        from_prop: str,
        from_value,
        to_label: str,
        to_prop: str,
        to_value,
        edge_type: str,
        edge_props: dict,
    ) -> str:
        """GQL: MATCH (a:<from_label>), (b:<to_label>) WHERE ... CREATE (a)-[:<edge_type>]->(b)"""
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
            return (
                f"MATCH (a:{from_label}), (b:{to_label}) "
                f"WHERE a.{from_prop} = {from_val} AND b.{to_prop} = {to_val} "
                f"CREATE (a)-[r:{edge_type} {{{prop_str}}}]->(b) RETURN r"
            )
        else:
            return (
                f"MATCH (a:{from_label}), (b:{to_label}) "
                f"WHERE a.{from_prop} = {from_val} AND b.{to_prop} = {to_val} "
                f"CREATE (a)-[r:{edge_type}]->(b) RETURN r"
            )

    def update_node_query(self, label: str, match_prop: str, match_value, set_prop: str, set_value) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> = <value> SET n.<prop> = <value>"""
        match_val = f"'{match_value}'" if isinstance(match_value, str) else match_value
        set_val = f"'{set_value}'" if isinstance(set_value, str) else set_value
        return (
            f"MATCH (n:{label}) WHERE n.{match_prop} = {match_val} "
            f"SET n.{set_prop} = {set_val} RETURN n"
        )


# =============================================================================
# GQL-SPECIFIC MUTATION TESTS
# =============================================================================

class TestGQLSpecificMutations:
    """GQL-specific mutation tests."""

    def test_gql_insert_syntax(self, db):
        """Test GQL INSERT syntax specifically."""
        result = db.execute("INSERT (:Person {name: 'InsertTest', age: 42}) RETURN *")
        rows = list(result)

        result = db.execute("MATCH (n:Person) WHERE n.name = 'InsertTest' RETURN n.age")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.age"] == 42

    def test_gql_multiple_labels_syntax(self, db):
        """Test GQL multiple labels syntax."""
        result = db.execute(
            "INSERT (:Person:Developer:Senior {name: 'MultiLabel'}) RETURN *"
        )
        list(result)

        result = db.execute("MATCH (n:Person:Developer) RETURN n.name")
        rows = list(result)
        assert len(rows) >= 1

    def test_gql_property_types(self, db):
        """Test various property types in GQL."""
        db.execute(
            "INSERT (:Data {str: 'hello', num: 42, flt: 3.14, bool: true})"
        )

        result = db.execute("MATCH (n:Data) RETURN n.str, n.num, n.flt, n.bool")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.str"] == "hello"
        assert rows[0]["n.num"] == 42

    def test_gql_set_multiple_properties(self, db):
        """Test SET with multiple properties."""
        db.execute("INSERT (:Person {name: 'Alice', age: 30, city: 'NYC'})")

        db.execute(
            "MATCH (n:Person) WHERE n.name = 'Alice' "
            "SET n.age = 31, n.city = 'LA' RETURN n"
        )

        result = db.execute("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.age, n.city")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.age"] == 31
        assert rows[0]["n.city"] == "LA"

    def test_gql_remove_property(self, db):
        """Test removing property by setting to null (REMOVE not yet in GQL)."""
        db.execute("INSERT (:Person {name: 'Bob', age: 25, temp: 'delete_me'})")

        # GQL uses SET n.prop = null to remove properties (REMOVE not yet implemented)
        db.execute(
            "MATCH (n:Person) WHERE n.name = 'Bob' "
            "SET n.temp = null RETURN n"
        )

        result = db.execute("MATCH (n:Person) WHERE n.name = 'Bob' RETURN n.temp")
        rows = list(result)
        # Property should be null or not present
        if len(rows) > 0:
            assert rows[0].get("n.temp") is None

    def test_gql_detach_delete(self, db):
        """Test DETACH DELETE (delete node and all relationships)."""
        alice = db.create_node(["Person"], {"name": "Alice"})
        bob = db.create_node(["Person"], {"name": "Bob"})
        db.create_edge(alice.id, bob.id, "KNOWS", {})

        # Delete Alice and her relationships
        db.execute(
            "MATCH (n:Person) WHERE n.name = 'Alice' DETACH DELETE n"
        )

        # Verify Alice is deleted
        result = db.execute("MATCH (n:Person) RETURN n.name")
        rows = list(result)
        names = [r["n.name"] for r in rows]
        assert "Alice" not in names
        assert "Bob" in names

    def test_gql_merge_create(self, db):
        """Test MERGE creates node if not exists."""
        db.execute(
            "MERGE (:Person {name: 'MergeTest'})"
        )

        result = db.execute("MATCH (n:Person) WHERE n.name = 'MergeTest' RETURN n.name")
        rows = list(result)
        assert len(rows) == 1

    def test_gql_merge_match(self, db):
        """Test MERGE matches existing node."""
        db.execute("INSERT (:Person {name: 'MergeExisting', age: 30})")

        # MERGE should match existing node, not create new
        db.execute(
            "MERGE (n:Person {name: 'MergeExisting'}) "
            "SET n.age = 31 RETURN n"
        )

        result = db.execute("MATCH (n:Person) WHERE n.name = 'MergeExisting' RETURN count(n) AS cnt")
        rows = list(result)
        assert rows[0]["cnt"] == 1

        result = db.execute("MATCH (n:Person) WHERE n.name = 'MergeExisting' RETURN n.age")
        rows = list(result)
        assert rows[0]["n.age"] == 31
