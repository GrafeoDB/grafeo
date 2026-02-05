"""Abstract base test classes for Grafeo test suite.

This module contains base classes that define WHAT to test. Each language
implementation inherits from these and provides HOW (the query syntax).

Base Test Classes:
- BaseQueriesTest: Pattern matching, paths, aggregations
- BaseMutationsTest: CRUD operations
- BaseTransactionsTest: Transaction handling
- BaseAlgorithmsTest: Graph algorithms

Base Benchmark Classes:
- BaseBenchStorage: Storage benchmarks (reads + writes)
- BaseBenchAlgorithms: Algorithm benchmarks

Comparison Test Classes:
- BaseNetworkXComparisonTest: Compare against NetworkX reference
- BaseNetworkXBenchmarkTest: Benchmark against NetworkX
- BaseSolvORComparisonTest: Compare against OR-Tools reference
- BaseSolvORBenchmarkTest: Benchmark against OR-Tools
"""

from .test_queries import BaseQueriesTest
from .test_mutations import BaseMutationsTest
from .test_transactions import BaseTransactionsTest
from .test_algorithms import BaseAlgorithmsTest

from .bench_storage import BaseBenchStorage, BenchmarkResult
from .bench_algorithms import BaseBenchAlgorithms

from .test_networkx import BaseNetworkXComparisonTest, BaseNetworkXBenchmarkTest
from .test_solvor import BaseSolvORComparisonTest, BaseSolvORBenchmarkTest

__all__ = [
    # Test base classes
    "BaseQueriesTest",
    "BaseMutationsTest",
    "BaseTransactionsTest",
    "BaseAlgorithmsTest",
    # Benchmark base classes
    "BaseBenchStorage",
    "BaseBenchAlgorithms",
    "BenchmarkResult",
    # Comparison test classes
    "BaseNetworkXComparisonTest",
    "BaseNetworkXBenchmarkTest",
    "BaseSolvORComparisonTest",
    "BaseSolvORBenchmarkTest",
]
