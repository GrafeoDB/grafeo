"""GQL implementation of storage benchmarks.

Benchmarks read and write operations using GQL syntax.
"""

import random
import pytest
from tests.python.bases.bench_storage import BaseBenchStorage

# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


class BenchGQLStorage(BaseBenchStorage):
    """GQL implementation of storage benchmarks."""

    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================

    def create_single_node(self, db, labels: list[str], props: dict):
        """Create a single node using GrafeoDB API."""
        return db.create_node(labels, props)

    def create_edge(self, db, source_id, target_id, rel_type: str, props: dict):
        """Create a single edge using GrafeoDB API."""
        return db.create_edge(source_id, target_id, rel_type, props)

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    def execute_query(self, db, query: str) -> list:
        """Execute a GQL query and return results as a list."""
        return list(db.execute(query))

    def full_scan_query(self, label: str, limit: int = None) -> str:
        """GQL: MATCH (n:Label) RETURN n [LIMIT x]"""
        query = f"MATCH (n:{label}) RETURN n"
        if limit:
            query += f" LIMIT {limit}"
        return query

    def count_query(self, label: str) -> str:
        """GQL: MATCH (n:Label) RETURN count(n)"""
        return f"MATCH (n:{label}) RETURN count(n)"

    def filter_query(self, label: str, prop: str, op: str, value) -> str:
        """GQL: MATCH (n:Label) WHERE n.prop op value RETURN n"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {value_str} RETURN n"

    def point_lookup_query(self, label: str, prop: str, value) -> str:
        """GQL: MATCH (n:Label) WHERE n.prop = value RETURN n"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} = {value_str} RETURN n"

    def one_hop_query(self, from_label: str, rel_type: str, to_label: str, limit: int = None) -> str:
        """GQL: MATCH (a:Label)-[:REL]->(b:Label) RETURN a, b"""
        query = f"MATCH (a:{from_label})-[:{rel_type}]->(b:{to_label}) RETURN a, b"
        if limit:
            query += f" LIMIT {limit}"
        return query

    def two_hop_query(self, label: str, rel_type: str, limit: int = None) -> str:
        """GQL: MATCH (a)-[:REL]->(b)-[:REL]->(c) RETURN count(c)"""
        return f"MATCH (a:{label})-[:{rel_type}]->(b:{label})-[:{rel_type}]->(c:{label}) RETURN count(c)"

    def aggregation_query(self, label: str, group_prop: str, agg_prop: str) -> str:
        """GQL: MATCH (n:Label) RETURN n.group_prop, count(n), avg(n.agg_prop)"""
        return (
            f"MATCH (n:{label}) "
            f"RETURN n.{group_prop}, count(n) AS cnt, avg(n.{agg_prop}) AS avg_{agg_prop} "
            f"ORDER BY cnt DESC"
        )

    def sort_query(self, label: str, sort_prop: str, desc: bool = False, limit: int = 100) -> str:
        """GQL: MATCH (n:Label) RETURN n ORDER BY n.prop [DESC] LIMIT x"""
        order = "DESC" if desc else "ASC"
        return f"MATCH (n:{label}) RETURN n ORDER BY n.{sort_prop} {order} LIMIT {limit}"

    def triangle_query(self, label: str, rel_type: str) -> str:
        """GQL: MATCH (a)-[:REL]->(b)-[:REL]->(c)-[:REL]->(a) RETURN count(a)"""
        return (
            f"MATCH (a:{label})-[:{rel_type}]->(b:{label})-[:{rel_type}]->(c:{label})-[:{rel_type}]->(a) "
            f"RETURN count(a)"
        )

    # =========================================================================
    # GRAPH SETUP
    # =========================================================================

    def setup_social_network(self, db, num_nodes: int, avg_edges: int):
        """Set up a social network graph for benchmarking."""
        random.seed(42)
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]

        nodes = []
        for i in range(num_nodes):
            node = db.create_node(["Person"], {
                "name": f"user{i}",
                "email": f"user{i}@example.com",
                "age": random.randint(18, 80),
                "city": random.choice(cities),
                "salary": random.uniform(30000, 150000),
            })
            nodes.append(node)

        total_edges = num_nodes * avg_edges
        for _ in range(total_edges):
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if src.id != dst.id:
                db.create_edge(src.id, dst.id, "KNOWS", {"since": random.randint(2000, 2024)})

    def setup_clique_graph(self, db, num_cliques: int, clique_size: int):
        """Set up a clique graph for triangle benchmarking."""
        random.seed(42)

        all_nodes = []
        for c in range(num_cliques):
            clique_nodes = []
            for i in range(clique_size):
                node = db.create_node(["Node"], {"clique": c, "idx": i})
                clique_nodes.append(node)
            all_nodes.extend(clique_nodes)

            for i, n1 in enumerate(clique_nodes):
                for n2 in clique_nodes[i + 1:]:
                    db.create_edge(n1.id, n2.id, "CONNECTED", {})
                    db.create_edge(n2.id, n1.id, "CONNECTED", {})

        for _ in range(num_cliques * 2):
            n1 = random.choice(all_nodes)
            n2 = random.choice(all_nodes)
            if n1.id != n2.id:
                db.create_edge(n1.id, n2.id, "CONNECTED", {})


# =============================================================================
# PYTEST FIXTURES AND TESTS
# =============================================================================

@pytest.fixture
def bench_suite():
    """Create a benchmark suite."""
    return BenchGQLStorage(warmup_iterations=2, iterations=3)


@pytest.fixture
def db_factory():
    """Factory for creating database instances."""
    if not GRAFEO_AVAILABLE:
        pytest.skip("Grafeo not installed")

    def create_db():
        return GrafeoDB()
    return create_db


class TestGQLStorageBenchmarks:
    """GQL storage benchmark tests.

    Run with: pytest tests/python/lpg/gql/bench_storage.py -v -m benchmark
    """

    @pytest.mark.benchmark
    def test_bench_single_node_insert(self, bench_suite, db_factory):
        """Benchmark single node insertion."""
        result = bench_suite.bench_single_node_insert(db_factory, count=500)
        assert result.mean_time_ms > 0

    @pytest.mark.benchmark
    def test_bench_edge_insert(self, bench_suite, db_factory):
        """Benchmark edge insertion."""
        result = bench_suite.bench_edge_insert(db_factory, node_count=50)
        assert result.mean_time_ms > 0

    @pytest.mark.benchmark
    def test_bench_full_scan(self, bench_suite, db_factory):
        """Benchmark full scan."""
        def setup(db):
            bench_suite.setup_social_network(db, 300, 3)
        result = bench_suite.bench_full_scan(db_factory, setup, 300)
        assert result.mean_time_ms > 0

    @pytest.mark.benchmark
    def test_bench_one_hop_traversal(self, bench_suite, db_factory):
        """Benchmark 1-hop traversal."""
        def setup(db):
            bench_suite.setup_social_network(db, 300, 3)
        result = bench_suite.bench_one_hop_traversal(db_factory, setup)
        assert result.mean_time_ms > 0

    @pytest.mark.benchmark
    def test_bench_aggregation(self, bench_suite, db_factory):
        """Benchmark aggregation."""
        def setup(db):
            bench_suite.setup_social_network(db, 300, 3)
        result = bench_suite.bench_aggregation(db_factory, setup)
        assert result.mean_time_ms > 0


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_benchmarks():
    """Run all benchmarks when called directly."""
    if not GRAFEO_AVAILABLE:
        print("ERROR: Grafeo not installed")
        return

    print("=" * 80)
    print("GQL STORAGE BENCHMARKS")
    print("=" * 80)

    suite = BenchGQLStorage(warmup_iterations=2, iterations=5)
    suite.run_all(GrafeoDB, node_count=500, avg_edges=5)


if __name__ == "__main__":
    run_benchmarks()
