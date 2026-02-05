"""Gremlin implementation of mutation tests.

Tests CRUD operations using Gremlin syntax.
"""

import pytest
from tests.python.bases.test_mutations import BaseMutationsTest


def execute_gremlin(db, query: str):
    """Execute Gremlin query, skip if not supported."""
    try:
        return db.execute_gremlin(query)
    except AttributeError:
        pytest.skip("Gremlin support not available in this build")
    except NotImplementedError:
        pytest.skip("Gremlin not implemented")


class TestGremlinMutations(BaseMutationsTest):
    """Gremlin implementation of mutation tests.

    Note: Uses g.addV(), g.addE(), drop() for mutations.
    """

    def create_node_query(self, labels: list[str], props: dict) -> str:
        """Return Gremlin addV query."""
        label = labels[0] if labels else "Vertex"
        prop_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                prop_parts.append(f".property('{k}', '{v}')")
            else:
                prop_parts.append(f".property('{k}', {v})")
        props_str = "".join(prop_parts)
        return f"g.addV('{label}'){props_str}"

    def delete_node_query(self, label: str, prop: str, value) -> str:
        """Return Gremlin drop query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"g.V().hasLabel('{label}').has('{prop}', {val}).drop()"

    def create_edge_query(self, from_label: str, from_prop: str, from_value,
                          to_label: str, to_prop: str, to_value,
                          edge_type: str, edge_props: dict) -> str:
        """Return Gremlin addE query."""
        from_val = f"'{from_value}'" if isinstance(from_value, str) else from_value
        to_val = f"'{to_value}'" if isinstance(to_value, str) else to_value
        prop_parts = []
        for k, v in edge_props.items():
            if isinstance(v, str):
                prop_parts.append(f".property('{k}', '{v}')")
            else:
                prop_parts.append(f".property('{k}', {v})")
        props_str = "".join(prop_parts)
        return (
            f"g.V().has('{from_prop}', {from_val})"
            f".addE('{edge_type}')"
            f".to(g.V().has('{to_prop}', {to_val}))"
            f"{props_str}"
        )

    def delete_edge_query(self, edge_type: str, from_prop: str, from_value,
                          to_prop: str, to_value) -> str:
        """Return Gremlin edge drop query."""
        from_val = f"'{from_value}'" if isinstance(from_value, str) else from_value
        to_val = f"'{to_value}'" if isinstance(to_value, str) else to_value
        return (
            f"g.V().has('{from_prop}', {from_val})"
            f".outE('{edge_type}')"
            f".where(inV().has('{to_prop}', {to_val}))"
            f".drop()"
        )

    def update_node_query(self, label: str, match_prop: str, match_value,
                          set_prop: str, set_value) -> str:
        """Return Gremlin property update query."""
        match_val = f"'{match_value}'" if isinstance(match_value, str) else match_value
        set_val = f"'{set_value}'" if isinstance(set_value, str) else set_value
        return (
            f"g.V().hasLabel('{label}').has('{match_prop}', {match_val})"
            f".property('{set_prop}', {set_val})"
        )

    def match_node_query(self, label: str, prop: str, value) -> str:
        """Return Gremlin match query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"g.V().hasLabel('{label}').has('{prop}', {val})"

    def count_nodes_query(self, label: str) -> str:
        """Return Gremlin count query."""
        return f"g.V().hasLabel('{label}').count()"

    def count_edges_query(self, edge_type: str) -> str:
        """Return Gremlin edge count query."""
        return f"g.E().hasLabel('{edge_type}').count()"

    def execute_query(self, db, query: str):
        """Execute query using execute_gremlin."""
        return execute_gremlin(db, query)


class TestGremlinMutationsDirect:
    """Gremlin-specific mutation tests using direct API."""

    def _execute_gremlin(self, db, query: str):
        """Execute Gremlin query, skip if not supported."""
        try:
            return db.execute_gremlin(query)
        except AttributeError:
            pytest.skip("Gremlin support not available")
        except NotImplementedError:
            pytest.skip("Gremlin not implemented")

    def test_gremlin_add_vertex(self, db):
        """Test g.addV() vertex creation."""
        result = self._execute_gremlin(db, "g.addV('Person').property('name', 'Alice')")
        rows = list(result)
        assert len(rows) >= 1

    def test_gremlin_add_vertex_multiple_props(self, db):
        """Test g.addV() with multiple properties."""
        result = self._execute_gremlin(
            db,
            "g.addV('Person').property('name', 'Bob').property('age', 25)"
        )
        rows = list(result)
        assert len(rows) >= 1

    def test_gremlin_add_edge(self, db):
        """Test g.addE() edge creation."""
        # Create vertices first
        self._execute_gremlin(db, "g.addV('Person').property('name', 'Alice')")
        self._execute_gremlin(db, "g.addV('Person').property('name', 'Bob')")

        # Add edge
        result = self._execute_gremlin(
            db,
            "g.V().has('name', 'Alice').addE('knows').to(g.V().has('name', 'Bob'))"
        )
        rows = list(result)
        assert len(rows) >= 1

    def test_gremlin_drop_vertex(self, db):
        """Test drop() vertex deletion."""
        self._execute_gremlin(db, "g.addV('Person').property('name', 'ToDelete')")

        # Verify exists
        result = self._execute_gremlin(db, "g.V().has('name', 'ToDelete').count()")
        rows = list(result)
        if rows:
            count = rows[0] if isinstance(rows[0], int) else rows[0].get("count", 0)
            assert count >= 1

        # Delete
        self._execute_gremlin(db, "g.V().has('name', 'ToDelete').drop()")

        # Verify deleted
        result = self._execute_gremlin(db, "g.V().has('name', 'ToDelete').count()")
        rows = list(result)
        if rows:
            count = rows[0] if isinstance(rows[0], int) else rows[0].get("count", 0)
            assert count == 0

    def test_gremlin_property_update(self, db):
        """Test property() update."""
        self._execute_gremlin(db, "g.addV('Person').property('name', 'Alice').property('age', 30)")

        # Update age
        self._execute_gremlin(db, "g.V().has('name', 'Alice').property('age', 31)")

        # Verify update
        result = self._execute_gremlin(db, "g.V().has('name', 'Alice').values('age')")
        rows = list(result)
        if rows:
            age = rows[0] if isinstance(rows[0], int) else rows[0].get("age", 0)
            assert age == 31
