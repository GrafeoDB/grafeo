"""
Grafeo - A high-performance, embeddable graph database.

This module provides Python bindings for the Grafeo graph database,
offering a Pythonic interface for graph operations and GQL queries.

Example:
    >>> from grafeo import GrafeoDB
    >>> db = GrafeoDB()
    >>> node = db.create_node(["Person"], {"name": "Alice", "age": 30})
    >>> result = db.execute("MATCH (n:Person) RETURN n")
    >>> for row in result:
    ...     print(row)
"""

from grafeo.grafeo import (
    GrafeoDB,
    Node,
    Edge,
    QueryResult,
    Value,
    __version__,
)

__all__ = [
    "GrafeoDB",
    "Node",
    "Edge",
    "QueryResult",
    "Value",
    "__version__",
]
