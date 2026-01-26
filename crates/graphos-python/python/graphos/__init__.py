"""
Graphos - A high-performance, embeddable graph database.

This module provides Python bindings for the Graphos graph database,
offering a Pythonic interface for graph operations and GQL queries.

Example:
    >>> from graphos import GraphosDB
    >>> db = GraphosDB()
    >>> node = db.create_node(["Person"], {"name": "Alice", "age": 30})
    >>> result = db.execute("MATCH (n:Person) RETURN n")
    >>> for row in result:
    ...     print(row)
"""

from graphos.graphos import (
    GraphosDB,
    Node,
    Edge,
    QueryResult,
    Value,
    __version__,
)

__all__ = [
    "GraphosDB",
    "Node",
    "Edge",
    "QueryResult",
    "Value",
    "__version__",
]
