"""
Grafeo - A high-performance, embeddable graph database.

This module provides Python bindings for the Grafeo graph database,
offering a Pythonic interface for graph operations and GQL queries.

Example:
    >>> from grafeo import GrafeoDB
    >>> db = GrafeoDB()
    >>> node = db.create_node(["Person"], {"name": "Alix", "age": 30})
    >>> result = db.execute("MATCH (n:Person) RETURN n")
    >>> for row in result:
    ...     print(row)
"""

from grafeo.grafeo import (
    Edge,
    GrafeoDB,
    GrafeoError,
    Node,
    QueryResult,
    ResultStream,
    Value,
    __version__,
    build_info,
    features,
    simd_support,
    vector,
)

__all__ = [
    "GrafeoDB",
    "GrafeoError",
    "Node",
    "Edge",
    "QueryResult",
    "ResultStream",
    "Value",
    "__version__",
    "build_info",
    "features",
    "simd_support",
    "vector",
]
