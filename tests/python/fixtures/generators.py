"""
Synthetic dataset generators for testing Grafeo.

This module provides various graph generators for testing and benchmarking:
- Social network graphs (scale-free)
- LDBC-like datasets
- Random graphs (Erdos-Renyi)
- Tree structures
- Dense cliques
"""

import random
import string
from dataclasses import dataclass
from typing import Iterator


@dataclass
class NodeData:
    """Represents a node to be inserted."""
    labels: list[str]
    properties: dict


@dataclass
class EdgeData:
    """Represents an edge to be inserted."""
    source_idx: int  # Index in the node list
    target_idx: int
    edge_type: str
    properties: dict


class SyntheticDataGenerator:
    """Base class for synthetic data generators."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.nodes: list[NodeData] = []
        self.edges: list[EdgeData] = []

    def random_string(self, length: int = 10) -> str:
        """Generate a random string."""
        return ''.join(self.rng.choices(string.ascii_lowercase, k=length))

    def random_name(self) -> str:
        """Generate a random name."""
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                       "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo",
                       "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rose"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                      "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        return f"{self.rng.choice(first_names)} {self.rng.choice(last_names)}"

    def generate(self) -> tuple[list[NodeData], list[EdgeData]]:
        """Generate the dataset. Override in subclasses."""
        raise NotImplementedError


class SocialNetworkGenerator(SyntheticDataGenerator):
    """
    Generate a social network graph with preferential attachment (scale-free).

    Creates Person nodes with KNOWS edges following a power-law degree distribution.
    """

    def __init__(
        self,
        num_nodes: int = 1000,
        avg_edges_per_node: int = 10,
        seed: int = 42,
    ):
        super().__init__(seed)
        self.num_nodes = num_nodes
        self.avg_edges_per_node = avg_edges_per_node

    def generate(self) -> tuple[list[NodeData], list[EdgeData]]:
        """Generate the social network."""
        # Generate Person nodes
        for i in range(self.num_nodes):
            self.nodes.append(NodeData(
                labels=["Person"],
                properties={
                    "name": self.random_name(),
                    "age": self.rng.randint(18, 80),
                    "city": self.rng.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"]),
                    "email": f"user{i}@example.com",
                }
            ))

        # Generate KNOWS edges using preferential attachment
        # Start with a small connected component
        if self.num_nodes >= 3:
            self.edges.append(EdgeData(0, 1, "KNOWS", {"since": self.rng.randint(2010, 2024)}))
            self.edges.append(EdgeData(1, 2, "KNOWS", {"since": self.rng.randint(2010, 2024)}))
            self.edges.append(EdgeData(2, 0, "KNOWS", {"since": self.rng.randint(2010, 2024)}))

        # Track degrees for preferential attachment
        degrees = [0] * self.num_nodes
        degrees[0] = 2
        degrees[1] = 2
        degrees[2] = 2

        # Add remaining edges with preferential attachment
        target_edges = self.num_nodes * self.avg_edges_per_node // 2
        existing_edges = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)}

        while len(self.edges) < target_edges:
            # Select source uniformly
            src = self.rng.randint(0, self.num_nodes - 1)

            # Select target with probability proportional to degree
            total_degree = sum(degrees)
            if total_degree == 0:
                dst = self.rng.randint(0, self.num_nodes - 1)
            else:
                r = self.rng.random() * total_degree
                cumsum = 0
                dst = 0
                for i, d in enumerate(degrees):
                    cumsum += d
                    if cumsum > r:
                        dst = i
                        break

            # Avoid self-loops and duplicates
            if src != dst and (src, dst) not in existing_edges:
                self.edges.append(EdgeData(
                    src, dst, "KNOWS",
                    {"since": self.rng.randint(2010, 2024)}
                ))
                existing_edges.add((src, dst))
                existing_edges.add((dst, src))
                degrees[src] += 1
                degrees[dst] += 1

        return self.nodes, self.edges


class LDBCLikeGenerator(SyntheticDataGenerator):
    """
    Generate an LDBC-like social network graph.

    Creates multiple entity types: Person, Company, University, City
    with various relationship types.
    """

    def __init__(
        self,
        scale_factor: float = 0.1,  # 0.1 = ~1K persons, 1.0 = ~10K persons
        seed: int = 42,
    ):
        super().__init__(seed)
        self.scale_factor = scale_factor
        self.num_persons = int(1000 * scale_factor)
        self.num_companies = int(100 * scale_factor)
        self.num_universities = int(50 * scale_factor)
        self.num_cities = int(20 * scale_factor)

    def generate(self) -> tuple[list[NodeData], list[EdgeData]]:
        """Generate the LDBC-like dataset."""
        # Generate Cities
        cities = ["New York", "San Francisco", "London", "Berlin", "Tokyo",
                  "Sydney", "Toronto", "Paris", "Amsterdam", "Singapore",
                  "Seattle", "Boston", "Austin", "Denver", "Miami",
                  "Chicago", "Atlanta", "Portland", "Dublin", "Munich"]
        for i in range(self.num_cities):
            city_name = cities[i % len(cities)]
            if i >= len(cities):
                city_name = f"{city_name} {i // len(cities) + 1}"
            self.nodes.append(NodeData(
                labels=["City"],
                properties={
                    "name": city_name,
                    "country": self.rng.choice(["USA", "UK", "Germany", "Japan", "Australia"]),
                }
            ))

        # Generate Universities
        uni_prefixes = ["University of", "Technical University", "MIT", "Stanford", "Harvard"]
        for i in range(self.num_universities):
            self.nodes.append(NodeData(
                labels=["University"],
                properties={
                    "name": f"{self.rng.choice(uni_prefixes)} {self.random_string(6).title()}",
                    "founded": self.rng.randint(1800, 2000),
                }
            ))

        # Generate Companies
        company_types = ["Inc", "Corp", "LLC", "Ltd", "GmbH"]
        industries = ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"]
        for i in range(self.num_companies):
            self.nodes.append(NodeData(
                labels=["Company"],
                properties={
                    "name": f"{self.random_string(8).title()} {self.rng.choice(company_types)}",
                    "industry": self.rng.choice(industries),
                    "founded": self.rng.randint(1950, 2020),
                }
            ))

        # Generate Persons
        for i in range(self.num_persons):
            self.nodes.append(NodeData(
                labels=["Person"],
                properties={
                    "name": self.random_name(),
                    "age": self.rng.randint(18, 80),
                    "email": f"user{i}@example.com",
                    "joined": self.rng.randint(2010, 2024),
                }
            ))

        # Calculate base indices
        # Cities: 0..num_cities
        # Universities: num_cities..num_cities+num_universities
        # Companies: num_cities+num_universities..num_cities+num_universities+num_companies
        # Persons: num_cities+num_universities+num_companies..end
        person_base = self.num_cities + self.num_universities + self.num_companies

        # Generate relationships
        # Person -[LIVES_IN]-> City
        for i in range(self.num_persons):
            person_idx = person_base + i
            city_idx = self.rng.randint(0, self.num_cities - 1)
            self.edges.append(EdgeData(person_idx, city_idx, "LIVES_IN", {}))

        # Person -[WORKS_AT]-> Company (70% of persons work)
        for i in range(self.num_persons):
            if self.rng.random() < 0.7:
                person_idx = person_base + i
                company_idx = self.num_cities + self.num_universities + self.rng.randint(0, self.num_companies - 1)
                self.edges.append(EdgeData(
                    person_idx, company_idx, "WORKS_AT",
                    {"since": self.rng.randint(2010, 2024)}
                ))

        # Person -[STUDIED_AT]-> University (60% of persons studied)
        for i in range(self.num_persons):
            if self.rng.random() < 0.6:
                person_idx = person_base + i
                uni_idx = self.num_cities + self.rng.randint(0, self.num_universities - 1)
                self.edges.append(EdgeData(
                    person_idx, uni_idx, "STUDIED_AT",
                    {"year": self.rng.randint(1990, 2020)}
                ))

        # Person -[KNOWS]-> Person (social connections)
        existing_knows = set()
        target_knows = self.num_persons * 5  # Average 5 friends per person

        while len(existing_knows) < target_knows:
            src = self.rng.randint(0, self.num_persons - 1)
            dst = self.rng.randint(0, self.num_persons - 1)
            if src != dst and (src, dst) not in existing_knows:
                self.edges.append(EdgeData(
                    person_base + src, person_base + dst, "KNOWS",
                    {"since": self.rng.randint(2010, 2024)}
                ))
                existing_knows.add((src, dst))

        # Company -[LOCATED_IN]-> City
        for i in range(self.num_companies):
            company_idx = self.num_cities + self.num_universities + i
            city_idx = self.rng.randint(0, self.num_cities - 1)
            self.edges.append(EdgeData(company_idx, city_idx, "LOCATED_IN", {}))

        # University -[LOCATED_IN]-> City
        for i in range(self.num_universities):
            uni_idx = self.num_cities + i
            city_idx = self.rng.randint(0, self.num_cities - 1)
            self.edges.append(EdgeData(uni_idx, city_idx, "LOCATED_IN", {}))

        return self.nodes, self.edges


class RandomGraphGenerator(SyntheticDataGenerator):
    """
    Generate an Erdos-Renyi random graph.

    Each pair of nodes has probability p of being connected.
    """

    def __init__(
        self,
        num_nodes: int = 1000,
        edge_probability: float = 0.01,
        seed: int = 42,
    ):
        super().__init__(seed)
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability

    def generate(self) -> tuple[list[NodeData], list[EdgeData]]:
        """Generate the random graph."""
        # Generate nodes
        for i in range(self.num_nodes):
            self.nodes.append(NodeData(
                labels=["Node"],
                properties={
                    "id": i,
                    "value": self.rng.random(),
                }
            ))

        # Generate edges with probability p
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.rng.random() < self.edge_probability:
                    self.edges.append(EdgeData(i, j, "CONNECTED", {}))

        return self.nodes, self.edges


class TreeGenerator(SyntheticDataGenerator):
    """
    Generate a tree structure.

    Useful for testing hierarchical queries.
    """

    def __init__(
        self,
        depth: int = 5,
        branching_factor: int = 3,
        seed: int = 42,
    ):
        super().__init__(seed)
        self.depth = depth
        self.branching_factor = branching_factor

    def generate(self) -> tuple[list[NodeData], list[EdgeData]]:
        """Generate the tree."""
        # Generate root
        self.nodes.append(NodeData(
            labels=["TreeNode", "Root"],
            properties={"level": 0, "name": "root"}
        ))

        # BFS to generate tree
        current_level = [0]
        for level in range(1, self.depth + 1):
            next_level = []
            for parent_idx in current_level:
                for _ in range(self.branching_factor):
                    child_idx = len(self.nodes)
                    self.nodes.append(NodeData(
                        labels=["TreeNode"],
                        properties={"level": level, "name": f"node_{child_idx}"}
                    ))
                    self.edges.append(EdgeData(parent_idx, child_idx, "PARENT_OF", {}))
                    next_level.append(child_idx)
            current_level = next_level

        return self.nodes, self.edges


class CliqueGenerator(SyntheticDataGenerator):
    """
    Generate a graph with dense cliques.

    Useful for testing triangle counting and community detection.
    """

    def __init__(
        self,
        num_cliques: int = 10,
        clique_size: int = 10,
        inter_clique_edges: int = 5,
        seed: int = 42,
    ):
        super().__init__(seed)
        self.num_cliques = num_cliques
        self.clique_size = clique_size
        self.inter_clique_edges = inter_clique_edges

    def generate(self) -> tuple[list[NodeData], list[EdgeData]]:
        """Generate the clique graph."""
        clique_starts = []

        # Generate cliques
        for c in range(self.num_cliques):
            start_idx = len(self.nodes)
            clique_starts.append(start_idx)

            # Generate nodes in clique
            for i in range(self.clique_size):
                self.nodes.append(NodeData(
                    labels=["Node", f"Clique{c}"],
                    properties={"clique": c, "local_id": i}
                ))

            # Generate all edges within clique (complete graph)
            for i in range(self.clique_size):
                for j in range(i + 1, self.clique_size):
                    self.edges.append(EdgeData(
                        start_idx + i, start_idx + j, "CONNECTED", {}
                    ))

        # Generate inter-clique edges
        for _ in range(self.inter_clique_edges * self.num_cliques):
            c1 = self.rng.randint(0, self.num_cliques - 1)
            c2 = self.rng.randint(0, self.num_cliques - 1)
            if c1 != c2:
                n1 = clique_starts[c1] + self.rng.randint(0, self.clique_size - 1)
                n2 = clique_starts[c2] + self.rng.randint(0, self.clique_size - 1)
                self.edges.append(EdgeData(n1, n2, "BRIDGE", {}))

        return self.nodes, self.edges


def load_data_into_db(db, generator: SyntheticDataGenerator) -> tuple[int, int]:
    """
    Load synthetic data into a Grafeo database.

    Returns tuple of (node_count, edge_count).
    """
    nodes, edges = generator.generate()

    # Insert nodes and track their IDs
    node_ids = []
    for node_data in nodes:
        node = db.create_node(node_data.labels, node_data.properties)
        node_ids.append(node.id)

    # Insert edges using the mapped IDs
    for edge_data in edges:
        src_id = node_ids[edge_data.source_idx]
        dst_id = node_ids[edge_data.target_idx]
        db.create_edge(src_id, dst_id, edge_data.edge_type, edge_data.properties)

    return len(nodes), len(edges)
