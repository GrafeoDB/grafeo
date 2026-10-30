"""Deriva-shaped selector and bulk-path benchmark.

Builds a graph shaped like Deriva's code graph (about 4.5k nodes: every node
`:Graph` plus one kind label, Methods with 31 properties, `active` flags and
paths, CONTAINS (File to Method, Package to File), USES, DEFINES,
HAS_METHOD, DEPENDS_ON, IN_ENV, COVERS and DEPRECATED_BY edges), then times the selector shapes and the Python bulk
paths, median of 10 runs after a warm-up.

Each query scenario first checks its result against a reference (the
hand-written UNION, or the same rows another way), so a faster wrong answer
fails the run instead of looking like a win.

Usage:
    python scripts/bench/deriva_selectors.py                 # print the table
    python scripts/bench/deriva_selectors.py --save base.json
    python scripts/bench/deriva_selectors.py --compare base.json

The guard: the EXISTS inside an OR branch whose other conjunct is false must
not cost more than GUARD_FACTOR times the same query without the EXISTS.
update_nodes must not cost more than UPDATE_FACTOR times the query it
replaces, and `MATCH (n) RETURN n` not more than RETURN_N_FACTOR times
get_nodes_by_label (Deriva's targets; skipped on a build without the call).
The selector-gap shapes (0.6.1 plan) each have a guard against a reference with
the same rows: the OR of labels under (n:Graph), labels(n) grouped, count(*)
of a label, an edge check on nodes with many edges of other types, and a
point lookup on two labels. With --compare, the guard rails (point lookups by
id, expansions, two hops, writes) fail at REGRESSION_FACTOR times the baseline.
"""

import argparse
import json
import random
import statistics
import sys
import time

import grafeo

KINDS = {
    "Method": 1500,
    "File": 800,
    "Class": 500,
    "Test": 600,
    "Config": 300,
    "Function": 300,
    "BuildModule": 200,
    "Document": 180,
    "Package": 100,
    "TypeDefinition": 150,
    "Environment": 20,
}
EXTRA_PROPERTIES = (
    28  # every node: id, name, path, kind, active, pagerank_percentile and these
)
RUNS = 10
GUARD_FACTOR = 3.0
UPDATE_FACTOR = 1.2  # update_nodes against the UNWIND ... MATCH ... SET query
RETURN_N_FACTOR = 1.5  # MATCH (n) RETURN n against get_nodes_by_label
# The selector-gap targets (0.6.1 plan): each selector shape against a reference
# that does the least work for the same rows.
OR_SCAN_FACTOR = (
    1.2  # an OR of labels under (n:Graph) against the UNION of the two labels
)
LABELS_COUNT_FACTOR = 1.5  # labels(n) grouped against a string property grouped
COUNT_FROM_STATS_FACTOR = 0.1  # count(*) of a label against reading its ids
TYPED_EXISTS_FACTOR = 1.5  # an edge check on busy nodes against reading their ids
MULTI_LABEL_SEEK_FACTOR = 1.3  # (n:Graph:Method {id}) against (n:Graph {id}), id index
# With --compare: what Grafeo already does well must not get slower.
GUARD_RAILS = [
    "point_lookups_by_id",
    "expand_one_hop",
    "two_hops",
    "read_ids",
    "kind_count",
    "same_or_without_exists",
    "get_nodes_by_label",
    "match_return_n",
    "update_nodes",
    "unwind_match_set",
    "upsert_nodes_with_id_index",
]
REGRESSION_FACTOR = 1.25


def build(seed=6):
    """The Deriva-shaped database and the node ids per kind."""
    rng = random.Random(seed)
    db = grafeo.GrafeoDB()
    ids = {}
    for kind, count in KINDS.items():
        rows = []
        for i in range(count):
            row = {
                "id": f"{kind.lower()}-{i}",
                "name": f"{kind}{i}",
                "path": f"src/{kind.lower()}s/m{i % 37}/{kind.lower()}_{i}.py",
                "kind": kind,
                "active": rng.random() < 0.7,
                "pagerank_percentile": rng.randint(0, 100),
            }
            for p in range(EXTRA_PROPERTIES):
                row[f"p{p}"] = (
                    rng.randint(0, 1000) if p % 2 else f"v{rng.randint(0, 50)}"
                )
            rows.append(row)
        ids[kind] = db.batch_create_nodes_with_props(["Graph", kind], rows)

    def link(src_kind, dst_kind, edge_type, per_source, probability=1.0):
        edges = []
        for src in ids[src_kind]:
            if rng.random() > probability:
                continue
            for _ in range(per_source):
                edges.append((src, rng.choice(ids[dst_kind]), edge_type))
        db.batch_create_edges(edges)

    link("File", "Method", "CONTAINS", 2)
    link("Method", "Method", "USES", 2, 0.8)
    link("File", "Class", "DEFINES", 1, 0.6)
    link("Class", "Method", "HAS_METHOD", 3)
    link("BuildModule", "BuildModule", "DEPENDS_ON", 2)
    link("BuildModule", "Environment", "IN_ENV", 1)
    link("Test", "Method", "COVERS", 2)
    link("Package", "File", "CONTAINS", 4)
    link("TypeDefinition", "Class", "DEFINES", 1, 0.5)
    link("Method", "Method", "DEPRECATED_BY", 1, 0.05)
    link("File", "File", "DEPRECATED_BY", 1, 0.05)
    return db, ids


def rows_of(result):
    return sorted(tuple(sorted(row.items())) for row in result)


# The selector shapes: (name, query, reference query giving the same rows).
SELECTORS = [
    (
        "or_of_labels",
        "MATCH (n:Graph) WHERE n.active AND (n:Method OR n:File) RETURN n.id AS id",
        "MATCH (n:Method) WHERE n.active RETURN n.id AS id UNION MATCH (n:File) WHERE n.active RETURN n.id AS id",
    ),
    (
        "deriva_selector_with_or",
        (
            "MATCH (n:Graph) WHERE (n:Method OR n:File) AND n.active "
            "AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
            "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } "
            "AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } RETURN n.id AS id"
        ),
        (
            "MATCH (n:Method) WHERE n.active "
            "AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
            "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } "
            "AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } RETURN n.id AS id "
            "UNION MATCH (n:File) WHERE n.active "
            "AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
            "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } "
            "AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } RETURN n.id AS id"
        ),
    ),
    (
        "deriva_selector_without_or",
        (
            "MATCH (n:Method) WHERE n.active "
            "AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
            "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } "
            "AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } RETURN n.id AS id"
        ),
        None,
    ),
    (
        "exists_in_or_with_false_conjunct",
        (
            "MATCH (n:Graph) WHERE (n.active = false AND n.active = true AND EXISTS { MATCH (n)-[:USES]->()-[:USES]->() }) "
            "OR n:File RETURN n.id AS id"
        ),
        "MATCH (n:File) RETURN n.id AS id",
    ),
    (
        "exists_in_or_unguarded",
        (
            "MATCH (n:Method) WHERE n.active "
            "OR EXISTS { MATCH (n)-[:USES]->()-[:USES]->(m) WHERE m.active } RETURN n.id AS id"
        ),
        (
            "MATCH (n:Method) WHERE n.active "
            "OR COUNT { MATCH (n)-[:USES]->()-[:USES]->(m) WHERE m.active } > 0 RETURN n.id AS id"
        ),
    ),
    (
        "exists_in_or_per_row",
        (
            "MATCH (n:Method) WHERE n.active "
            "OR COUNT { MATCH (n)-[:USES]->()-[:USES]->(m) WHERE m.active } > 0 RETURN n.id AS id"
        ),
        None,
    ),
    (
        "same_or_without_exists",
        "MATCH (n:Graph) WHERE (n.active = false AND n.active = true) OR n:File RETURN n.id AS id",
        "MATCH (n:File) RETURN n.id AS id",
    ),
    (
        "multi_label_exists",
        "MATCH (n:Graph:Method) WHERE EXISTS { MATCH (n)-[:USES]->(:Graph:Method) } RETURN n.id AS id",
        "MATCH (n:Method) WHERE EXISTS { MATCH (n)-[:USES]->(m:Method) } RETURN n.id AS id",
    ),
]


def median_ms(func, runs=RUNS):
    func()  # warm-up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def run():
    db, _ids = build()
    results, failures = {}, []
    for name, query, reference in SELECTORS:
        got = rows_of(db.execute(query))
        if reference is not None:
            want = rows_of(db.execute(reference))
            if got != want:
                failures.append(f"{name}: {len(got)} rows, the reference {len(want)}")
        results[name] = {
            "ms": median_ms(lambda q=query: db.execute(q)),
            "rows": len(got),
        }

    # The selector-gap shapes (0.6.1 plan), each checked against a reference.
    or_selector = (
        "MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND n.pagerank_percentile >= 90 "
        "RETURN n.id AS id"
    )
    or_union = (
        "MATCH (n:File) WHERE n.pagerank_percentile >= 90 RETURN n.id AS id "
        "UNION MATCH (n:TypeDefinition) WHERE n.pagerank_percentile >= 90 RETURN n.id AS id"
    )
    labels_count = "MATCH (n:Graph) RETURN labels(n) AS l, count(*) AS c"
    kind_count = "MATCH (n:Graph) RETURN n.kind AS k, count(*) AS c"
    count_star = "MATCH (n:Graph) RETURN count(*) AS c"
    read_ids = "MATCH (n:Graph) RETURN n.id AS id"
    typed_exists = "MATCH (n:BuildModule) WHERE EXISTS { MATCH (n)-[:IN_ENV]->() } RETURN n.id AS id"
    busy_ids = "MATCH (n:BuildModule) RETURN n.id AS id"
    if rows_of(db.execute(or_selector)) != rows_of(db.execute(or_union)):
        failures.append("or_of_labels_under_graph: not the rows of the UNION")
    label_counts = sorted(row["c"] for row in db.execute(labels_count))
    kind_counts = sorted(row["c"] for row in db.execute(kind_count))
    if label_counts != kind_counts:
        failures.append(f"labels_count: {label_counts}, by kind {kind_counts}")
    if db.execute(count_star).to_list()[0]["c"] != sum(KINDS.values()):
        failures.append("count_star: not the number of Graph nodes")
    for name, query in [
        ("or_of_labels_under_graph", or_selector),
        ("or_of_labels_union", or_union),
        ("labels_count", labels_count),
        ("kind_count", kind_count),
        ("type_count", "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c"),
        ("count_star", count_star),
        ("read_ids", read_ids),
        ("typed_exists_on_busy_nodes", typed_exists),
        ("busy_ids", busy_ids),
    ]:
        results[name] = {"ms": median_ms(lambda q=query: list(db.execute(q)))}

    # Python bulk paths against the query path.
    method_rows = [
        {"id": f"method-{i}", "p0": "updated", "p1": i} for i in range(KINDS["Method"])
    ]
    set_query = "UNWIND $rows AS r MATCH (n:Graph:Method {id: r.id}) SET n.p0 = r.p0, n.p1 = r.p1"
    results["upsert_nodes"] = {
        "ms": median_ms(lambda: db.upsert_nodes(["Graph", "Method"], method_rows))
    }
    results["unwind_match_set"] = {
        "ms": median_ms(lambda: db.execute(set_query, {"rows": method_rows}))
    }
    if hasattr(db, "update_nodes"):
        results["update_nodes"] = {
            "ms": median_ms(lambda: db.update_nodes(["Graph", "Method"], method_rows))
        }
    results["get_nodes_by_label"] = {
        "ms": median_ms(lambda: db.get_nodes_by_label("Method"))
    }
    method_ids = [node_id for node_id, _ in db.get_nodes_by_label("Method")]
    keys = ["id", "name", "path"] + [f"p{p}" for p in range(7)]
    if hasattr(db, "get_properties_batch"):
        results["get_properties_batch_10"] = {
            "ms": median_ms(lambda: db.get_properties_batch(method_ids, keys))
        }
    results["get_property_batch_x10"] = {
        "ms": median_ms(
            lambda: [db.get_property_batch(method_ids, key) for key in keys]
        )
    }
    results["match_return_n"] = {
        "ms": median_ms(lambda: list(db.execute("MATCH (n:Method) RETURN n")))
    }
    columns = ", ".join(
        ["n.id AS id", "n.name AS name", "n.path AS path"]
        + [f"n.p{p} AS p{p}" for p in range(EXTRA_PROPERTIES)]
    )
    results["named_columns_31"] = {
        "ms": median_ms(lambda: list(db.execute(f"MATCH (n:Method) RETURN {columns}")))
    }

    # Last, as the index changes the plans of everything after it.
    db.create_property_index("id")
    results["upsert_nodes_with_id_index"] = {
        "ms": median_ms(lambda: db.upsert_nodes(["Graph", "Method"], method_rows))
    }
    lookup_ids = [f"method-{i}" for i in range(0, KINDS["Method"], 15)]

    def each_id(query):
        return lambda: [list(db.execute(query, {"id": i})) for i in lookup_ids]

    results["point_lookups_by_id"] = {
        "ms": median_ms(each_id("MATCH (n:Graph {id: $id}) RETURN n.name AS name"))
    }
    results["multi_label_point_lookups"] = {
        "ms": median_ms(
            each_id("MATCH (n:Graph:Method {id: $id}) RETURN n.name AS name")
        )
    }
    results["expand_one_hop"] = {
        "ms": median_ms(
            each_id("MATCH (n:Method {id: $id})-[:USES]->(m) RETURN m.id AS id")
        )
    }
    two_hops = "MATCH (a:File)-[:CONTAINS]->(:Method)-[:USES]->(m) RETURN count(m) AS c"
    results["two_hops"] = {"ms": median_ms(lambda: list(db.execute(two_hops)))}

    # Deriva's targets: a dedicated bulk call is at least as fast as the query
    # it replaces, and whole nodes through the query path cost about what the
    # direct API costs.
    for name, against, limit in [
        ("update_nodes", "unwind_match_set", UPDATE_FACTOR),
        ("match_return_n", "get_nodes_by_label", RETURN_N_FACTOR),
        ("or_of_labels_under_graph", "or_of_labels_union", OR_SCAN_FACTOR),
        ("labels_count", "kind_count", LABELS_COUNT_FACTOR),
        ("count_star", "read_ids", COUNT_FROM_STATS_FACTOR),
        ("typed_exists_on_busy_nodes", "busy_ids", TYPED_EXISTS_FACTOR),
        ("multi_label_point_lookups", "point_lookups_by_id", MULTI_LABEL_SEEK_FACTOR),
    ]:
        if name in results:
            ratio = results[name]["ms"] / max(results[against]["ms"], 1e-6)
            if ratio > limit:
                failures.append(f"{name}: {ratio:.1f}x {against} (limit {limit}x)")

    guard = results["exists_in_or_with_false_conjunct"]["ms"] / max(
        results["same_or_without_exists"]["ms"], 1e-6
    )
    if guard > GUARD_FACTOR:
        failures.append(
            f"EXISTS short-circuit guard: {guard:.1f}x the query without the EXISTS (limit {GUARD_FACTOR}x)"
        )
    return results, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save")
    parser.add_argument("--compare")
    args = parser.parse_args()
    results, failures = run()
    base = {}
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            base = json.load(f)
    print(f"{'scenario':38} {'median ms':>10} {'baseline':>10} {'ratio':>7}  rows")
    for name, r in results.items():
        b = base.get(name, {}).get("ms")
        ratio = f"{r['ms'] / b:6.2f}x" if b else ""
        print(
            f"{name:38} {r['ms']:10.2f} {b if b is not None else '':>10} {ratio:>7}  {r.get('rows', '')}"
        )
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    for name in GUARD_RAILS:
        b = base.get(name, {}).get("ms")
        if b and name in results and results[name]["ms"] > REGRESSION_FACTOR * b:
            ratio = results[name]["ms"] / b
            failures.append(
                f"{name}: {ratio:.2f}x the baseline (limit {REGRESSION_FACTOR}x)"
            )
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
