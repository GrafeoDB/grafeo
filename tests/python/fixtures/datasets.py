"""Pre-built test graph datasets.

This module provides functions to create common test graphs.
"""

import random

from .generators import (
    SocialNetworkGenerator,
    LDBCLikeGenerator,
    TreeGenerator,
    CliqueGenerator,
    load_data_into_db,
)


def create_social_graph(db, size: int = 50, avg_edges: int = 5, seed: int = 42):
    """Create a social network graph.

    Args:
        db: Database instance
        size: Number of person nodes
        avg_edges: Average edges per node
        seed: Random seed for reproducibility

    Returns:
        dict with 'node_count', 'edge_count'
    """
    gen = SocialNetworkGenerator(num_nodes=size, avg_edges_per_node=avg_edges, seed=seed)
    node_count, edge_count = load_data_into_db(db, gen)
    return {"node_count": node_count, "edge_count": edge_count}


def create_ldbc_graph(db, scale: float = 0.1, seed: int = 42):
    """Create an LDBC-like graph with multiple entity types.

    Args:
        db: Database instance
        scale: Scale factor (0.1 = ~100 persons)
        seed: Random seed for reproducibility

    Returns:
        dict with 'node_count', 'edge_count'
    """
    gen = LDBCLikeGenerator(scale_factor=scale, seed=seed)
    node_count, edge_count = load_data_into_db(db, gen)
    return {"node_count": node_count, "edge_count": edge_count}


def create_tree_graph(db, depth: int = 3, branching: int = 2, seed: int = 42):
    """Create a tree structure.

    Args:
        db: Database instance
        depth: Tree depth
        branching: Branching factor
        seed: Random seed for reproducibility

    Returns:
        dict with 'node_count', 'edge_count'
    """
    gen = TreeGenerator(depth=depth, branching_factor=branching, seed=seed)
    node_count, edge_count = load_data_into_db(db, gen)
    return {"node_count": node_count, "edge_count": edge_count}


def create_clique_graph(db, num_cliques: int = 5, clique_size: int = 5, seed: int = 42):
    """Create a graph with dense cliques.

    Args:
        db: Database instance
        num_cliques: Number of cliques
        clique_size: Nodes per clique
        seed: Random seed for reproducibility

    Returns:
        dict with 'node_count', 'edge_count'
    """
    gen = CliqueGenerator(num_cliques=num_cliques, clique_size=clique_size, seed=seed)
    node_count, edge_count = load_data_into_db(db, gen)
    return {"node_count": node_count, "edge_count": edge_count}


def create_algorithm_test_graph(db, n_nodes: int = 100, n_edges: int = 300, seed: int = 42):
    """Create a random graph for algorithm testing.

    Args:
        db: Database instance
        n_nodes: Number of nodes
        n_edges: Number of edges
        seed: Random seed for reproducibility

    Returns:
        dict with 'node_ids' list and metadata
    """
    rng = random.Random(seed)

    node_ids = []
    for i in range(n_nodes):
        node = db.create_node(["Node"], {"index": i})
        node_ids.append(node.id)

    edges = set()
    while len(edges) < n_edges:
        src = rng.choice(node_ids)
        dst = rng.choice(node_ids)
        if src != dst and (src, dst) not in edges:
            db.create_edge(src, dst, "EDGE", {"weight": rng.uniform(0.1, 10.0)})
            edges.add((src, dst))

    return {
        "node_ids": node_ids,
        "edge_count": len(edges),
    }


def create_pattern_test_graph(db):
    """Create a graph for pattern matching tests.

    Creates:
    - Person nodes: Alice (30, NYC), Bob (25, LA), Charlie (35, NYC)
    - Company nodes: Acme Corp, Globex Inc
    - KNOWS edges between persons
    - WORKS_AT edges from persons to companies

    Returns:
        dict with node references
    """
    # Create Person nodes
    alice = db.create_node(["Person"], {
        "name": "Alice", "age": 30, "city": "NYC"
    })
    bob = db.create_node(["Person"], {
        "name": "Bob", "age": 25, "city": "LA"
    })
    charlie = db.create_node(["Person"], {
        "name": "Charlie", "age": 35, "city": "NYC"
    })

    # Create Company nodes
    acme = db.create_node(["Company"], {
        "name": "Acme Corp", "founded": 2010
    })
    globex = db.create_node(["Company"], {
        "name": "Globex Inc", "founded": 2015
    })

    # Create KNOWS edges
    db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})
    db.create_edge(bob.id, charlie.id, "KNOWS", {"since": 2021})
    db.create_edge(alice.id, charlie.id, "KNOWS", {"since": 2019})

    # Create WORKS_AT edges
    db.create_edge(alice.id, acme.id, "WORKS_AT", {"role": "Engineer"})
    db.create_edge(bob.id, globex.id, "WORKS_AT", {"role": "Manager"})
    db.create_edge(charlie.id, acme.id, "WORKS_AT", {"role": "Director"})

    return {
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "acme": acme,
        "globex": globex,
    }


def create_chain_graph(db):
    """Create a chain graph: a -> b -> c -> d.

    Returns:
        dict with node IDs
    """
    a = db.create_node(["Node"], {"name": "a"})
    b = db.create_node(["Node"], {"name": "b"})
    c = db.create_node(["Node"], {"name": "c"})
    d = db.create_node(["Node"], {"name": "d"})

    db.create_edge(a.id, b.id, "NEXT", {})
    db.create_edge(b.id, c.id, "NEXT", {})
    db.create_edge(c.id, d.id, "NEXT", {})

    return {"a": a.id, "b": b.id, "c": c.id, "d": d.id}


def create_multi_path_graph(db):
    """Create a graph with multiple paths between a and d.

    Creates:
    - Direct path: a -> d
    - Longer path: a -> b -> c -> d

    Returns:
        dict with node IDs
    """
    a = db.create_node(["Node"], {"name": "a"})
    b = db.create_node(["Node"], {"name": "b"})
    c = db.create_node(["Node"], {"name": "c"})
    d = db.create_node(["Node"], {"name": "d"})

    # Direct path
    db.create_edge(a.id, d.id, "DIRECT", {})

    # Longer path
    db.create_edge(a.id, b.id, "STEP", {})
    db.create_edge(b.id, c.id, "STEP", {})
    db.create_edge(c.id, d.id, "STEP", {})

    return {"a": a.id, "b": b.id, "c": c.id, "d": d.id}
