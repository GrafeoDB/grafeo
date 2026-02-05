"""GraphQL implementation of mutation tests.

Tests CRUD operations using GraphQL syntax.
"""

import pytest
from tests.python.bases.test_mutations import BaseMutationsTest


def execute_graphql(db, query: str):
    """Execute GraphQL query, skip if not supported."""
    try:
        return db.execute_graphql(query)
    except AttributeError:
        pytest.skip("GraphQL support not available in this build")
    except NotImplementedError:
        pytest.skip("GraphQL not implemented")


class TestGraphQLMutations(BaseMutationsTest):
    """GraphQL implementation of mutation tests.

    Note: Uses GraphQL mutations for CRUD operations.
    """

    def create_node_query(self, labels: list[str], props: dict) -> str:
        """Return GraphQL createNode mutation."""
        label = labels[0] if labels else "Node"
        args = ", ".join(
            f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
            for k, v in props.items()
        )
        return f"""
            mutation {{
                create{label}({args}) {{
                    id
                }}
            }}
        """

    def delete_node_query(self, label: str, prop: str, value) -> str:
        """Return GraphQL deleteNode mutation."""
        val = f'"{value}"' if isinstance(value, str) else value
        return f"""
            mutation {{
                delete{label}({prop}: {val}) {{
                    success
                }}
            }}
        """

    def create_edge_query(self, from_label: str, from_prop: str, from_value,
                          to_label: str, to_prop: str, to_value,
                          edge_type: str, edge_props: dict) -> str:
        """Return GraphQL createEdge mutation."""
        from_val = f'"{from_value}"' if isinstance(from_value, str) else from_value
        to_val = f'"{to_value}"' if isinstance(to_value, str) else to_value
        props_args = ", ".join(
            f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
            for k, v in edge_props.items()
        )
        return f"""
            mutation {{
                createEdge(
                    from{from_label}: {{ {from_prop}: {from_val} }}
                    to{to_label}: {{ {to_prop}: {to_val} }}
                    type: "{edge_type}"
                    {props_args}
                ) {{
                    id
                }}
            }}
        """

    def delete_edge_query(self, edge_type: str, from_prop: str, from_value,
                          to_prop: str, to_value) -> str:
        """Return GraphQL deleteEdge mutation."""
        from_val = f'"{from_value}"' if isinstance(from_value, str) else from_value
        to_val = f'"{to_value}"' if isinstance(to_value, str) else to_value
        return f"""
            mutation {{
                deleteEdge(
                    type: "{edge_type}"
                    from: {{ {from_prop}: {from_val} }}
                    to: {{ {to_prop}: {to_val} }}
                ) {{
                    success
                }}
            }}
        """

    def update_node_query(self, label: str, match_prop: str, match_value,
                          set_prop: str, set_value) -> str:
        """Return GraphQL updateNode mutation."""
        match_val = f'"{match_value}"' if isinstance(match_value, str) else match_value
        set_val = f'"{set_value}"' if isinstance(set_value, str) else set_value
        return f"""
            mutation {{
                update{label}({match_prop}: {match_val}, {set_prop}: {set_val}) {{
                    {set_prop}
                }}
            }}
        """

    def match_node_query(self, label: str, prop: str, value) -> str:
        """Return GraphQL query."""
        val = f'"{value}"' if isinstance(value, str) else value
        return f"""
            query {{
                {label.lower()}({prop}: {val}) {{
                    id
                }}
            }}
        """

    def count_nodes_query(self, label: str) -> str:
        """Return GraphQL count query."""
        return f"""
            query {{
                {label.lower()}Count
            }}
        """

    def count_edges_query(self, edge_type: str) -> str:
        """Return GraphQL edge count query."""
        return f"""
            query {{
                edgeCount(type: "{edge_type}")
            }}
        """

    def execute_query(self, db, query: str):
        """Execute query using execute_graphql."""
        return execute_graphql(db, query)
