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
    "Environment": 20,
}
EXTRA_PROPERTIES = 28  # a Method has id, name, path and these: 31
RUNS = 10
GUARD_FACTOR = 3.0
UPDATE_FACTOR = 1.2  # update_nodes against the UNWIND ... MATCH ... SET query
RETURN_N_FACTOR = 1.5  # MATCH (n) RETURN n against get_nodes_by_label


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
                "active": rng.random() < 0.7,
            }
            if kind == "Method":
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

    # Deriva's targets: a dedicated bulk call is at least as fast as the query
    # it replaces, and whole nodes through the query path cost about what the
    # direct API costs.
    for name, against, limit in [
        ("update_nodes", "unwind_match_set", UPDATE_FACTOR),
        ("match_return_n", "get_nodes_by_label", RETURN_N_FACTOR),
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
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
