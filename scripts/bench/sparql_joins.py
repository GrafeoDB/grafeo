"""SPARQL join benchmark: triple pattern joins and LDBC short-read shapes on RDF.

Builds a synthetic RDF graph (no external data): 20,000 subjects with `v:a` and
`v:b`, every fourth with `v:d` naming the next subject's `v:a` value; 1,000
people with a profile and a city; 1,000 posts in 100 moderated forums; 4,000
comments that reply to a post or to an earlier comment, in chains of up to four.
Then times, median of 10 runs after a warm-up:

- the object join `?s v:a ?x . ?t v:d ?x` in both written orders, and the
  subject join `?s v:a ?x . ?s v:b ?y` of 20k x 20k;
- `VALUES ?s { N subjects } ?s v:a ?x` for N 10, 1,024 and 1,025 (around the
  most rows a pattern is looked up for, row by row);
- IS1 (a person's profile and city), IS4 (a comment's date and content) and IS6
  (the forum and moderator of a comment's post, through `v:REPLY_OF*`), each
  over 10 bound ids;
- write-then-read: one INSERT DATA, then one IS4, ten times.

Each scenario first checks its answer, so a faster wrong answer fails the run.

Usage:
    python scripts/bench/sparql_joins.py                 # print the table
    python scripts/bench/sparql_joins.py --save base.json
    python scripts/bench/sparql_joins.py --compare base.json

The guards, within one run: the two written orders of the object join within
JOIN_ORDER_FACTOR of each other (the join hashes the smaller side either way);
the 1,024-row VALUES at most CROSSOVER_FACTOR times the 1,025-row one (1,024
rows are looked up, and the lookups are clearly cheaper than the hash join of
1,025: a limit set too low hashes both, one set too high makes the lookups the
slower); write-then-read at most WRITE_READ_FACTOR times IS4 (a write does not
make the next query collect the statistics again). The two scenarios of each
guarded pair are timed in turns, PAIR_RUNS each, so the machine's drift during
the run falls on both.
With --compare, the guard rails (the joins, the 1,024 and 1,025-row VALUES and
the three short reads) fail at REGRESSION_FACTOR times the baseline.
"""

import argparse
import json
import random
import statistics
import sys
import time

import grafeo

V = "http://bench.example/v/"
SUBJECTS = 20_000
PEOPLE = 1_000
CITIES = 20
POSTS = 1_000
FORUMS = 100
COMMENTS = 4_000
RUNS = 10
PAIR_RUNS = 21
JOIN_ORDER_FACTOR = 1.3
CROSSOVER_FACTOR = 0.9
WRITE_READ_FACTOR = 5.0
REGRESSION_FACTOR = 1.25
GUARD_RAILS = [
    "object_join_a_first",
    "object_join_d_first",
    "subject_join_20k",
    "values_1024",
    "values_1025",
    "is1",
    "is4",
    "is6",
]


def iri(kind, i):
    return f"<{V}{kind}{i}>"


def triples():
    """The graph's triples as N-Triples lines (without the final dot)."""
    out = []
    for i in range(SUBJECTS):
        s = iri("s", i)
        out.append(f'{s} <{V}a> "a{i}"')
        out.append(f'{s} <{V}b> "b{i}"')
        if i % 4 == 0 and i + 1 < SUBJECTS:
            out.append(f'{s} <{V}d> "a{i + 1}"')
    for k in range(CITIES):
        out.append(f'{iri("city", k)} <{V}id> "c{k}"')
    for i in range(PEOPLE):
        p = iri("person", i)
        out += [
            f"{p} a <{V}Person>",
            f'{p} <{V}id> "p{i}"',
            f'{p} <{V}firstName> "first{i}"',
            f'{p} <{V}lastName> "last{i}"',
            f"{p} <{V}birthday> {19_900_000 + i}",
            f'{p} <{V}browserUsed> "browser{i % 5}"',
            f"{p} <{V}LIVES_IN> {iri('city', i % CITIES)}",
        ]
    for f in range(FORUMS):
        out.append(f"{iri('forum', f)} <{V}HAS_MODERATOR> {iri('person', f)}")
    for i in range(POSTS):
        post = iri("post", i)
        out += [
            f"{post} a <{V}Post>",
            f"{iri('forum', i % FORUMS)} <{V}CONTAINER_OF> {post}",
        ]
    for i in range(COMMENTS):
        c = iri("comment", i)
        target = iri("post", i) if i < POSTS else iri("comment", i - POSTS)
        out += [
            f"{c} a <{V}Comment>",
            f'{c} <{V}id> "m{i}"',
            f"{c} <{V}creationDate> {1_000_000 + i}",
            f'{c} <{V}content> "text {i}"',
            f"{c} <{V}REPLY_OF> {target}",
        ]
    return out


def build():
    db = grafeo.GrafeoDB()
    lines = triples()
    for i in range(0, len(lines), 5_000):
        db.execute_sparql("INSERT DATA { " + " .\n".join(lines[i : i + 5_000]) + " . }")
    return db


def median_ms(func, runs=RUNS):
    func()  # warm-up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def paired_ms(first, second, runs=PAIR_RUNS):
    """The medians of `first` and `second`, run in turns after a warm-up."""
    first()
    second()
    times = ([], [])
    for _ in range(runs):
        for func, out in zip((first, second), times):
            start = time.perf_counter()
            func()
            out.append((time.perf_counter() - start) * 1000)
    return statistics.median(times[0]), statistics.median(times[1])


def count(db, query):
    return int(db.execute_sparql(query).to_list()[0]["n"])


def values(n):
    return "VALUES ?s { " + " ".join(iri("s", i) for i in range(n)) + " }"


def is1(i):
    return (
        f'SELECT ?first ?last ?birthday ?browser ?city WHERE {{ ?p <{V}id> "p{i}" ; '
        f"a <{V}Person> ; <{V}firstName> ?first ; <{V}lastName> ?last ; "
        f"<{V}birthday> ?birthday ; <{V}browserUsed> ?browser ; <{V}LIVES_IN> ?c . "
        f"?c <{V}id> ?city }}"
    )


def is4(i):
    return (
        f'SELECT ?date ?content WHERE {{ ?m <{V}id> "m{i}" ; a <{V}Comment> ; '
        f"<{V}creationDate> ?date ; <{V}content> ?content }}"
    )


def is6(i):
    return (
        f'SELECT ?forum ?moderator WHERE {{ VALUES ?mid {{ "m{i}" }} ?m <{V}id> ?mid ; '
        f"<{V}REPLY_OF>* ?post . ?post a <{V}Post> . "
        f"?forum <{V}CONTAINER_OF> ?post ; <{V}HAS_MODERATOR> ?moderator . }}"
    )


def run():
    db = build()
    results, failures = {}, []

    def checked(name, query, want):
        got = count(db, query)
        if got != want:
            failures.append(f"{name}: {got} rows, want {want}")
        return lambda: db.execute_sparql(query).to_list()

    def scenario(name, query, want):
        results[name] = {"ms": median_ms(checked(name, query, want))}

    def pair(names, queries, want):
        first = checked(names[0], queries[0], want[0])
        second = checked(names[1], queries[1], want[1])
        for name, ms in zip(names, paired_ms(first, second)):
            results[name] = {"ms": ms}

    quarter = len(range(0, SUBJECTS - 1, 4))
    pair(
        ("object_join_a_first", "object_join_d_first"),
        (
            f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s <{V}a> ?x . ?t <{V}d> ?x }}",
            f"SELECT (COUNT(*) AS ?n) WHERE {{ ?t <{V}d> ?x . ?s <{V}a> ?x }}",
        ),
        (quarter, quarter),
    )
    scenario(
        "subject_join_20k",
        f"SELECT (COUNT(*) AS ?n) WHERE {{ ?s <{V}a> ?x . ?s <{V}b> ?y }}",
        SUBJECTS,
    )
    scenario(
        "values_10",
        f"SELECT (COUNT(*) AS ?n) WHERE {{ {values(10)} ?s <{V}a> ?x }}",
        10,
    )
    pair(
        ("values_1024", "values_1025"),
        tuple(
            f"SELECT (COUNT(*) AS ?n) WHERE {{ {values(n)} ?s <{V}a> ?x }}"
            for n in (1_024, 1_025)
        ),
        (1_024, 1_025),
    )

    rng = random.Random(1)
    people = rng.sample(range(PEOPLE), 10)
    comments = rng.sample(range(COMMENTS), 10)
    for i in people:
        rows = db.execute_sparql(is1(i)).to_list()
        if len(rows) != 1 or str(rows[0]["first"]) != f"first{i}":
            failures.append(f"is1 p{i}: {rows[:2]}")
    for i in comments:
        rows = db.execute_sparql(is4(i)).to_list()
        if len(rows) != 1 or str(rows[0]["content"]) != f"text {i}":
            failures.append(f"is4 m{i}: {rows[:2]}")
        post = i % POSTS
        want = (f"{V}forum{post % FORUMS}", f"{V}person{post % FORUMS}")
        rows = db.execute_sparql(is6(i)).to_list()
        got = [(str(r["forum"]), str(r["moderator"])) for r in rows]
        if got != [want]:
            failures.append(f"is6 m{i}: {got[:2]}, want {want}")
    for name, form, ids in (
        ("is1", is1, people),
        ("is4", is4, comments),
        ("is6", is6, comments),
    ):
        results[name] = {
            "ms": median_ms(lambda f=form, s=ids: [db.execute_sparql(f(i)) for i in s])
            / len(ids)
        }

    serial = [0]

    def write_then_read():
        for i in comments:
            serial[0] += 1
            db.execute_sparql(
                f'INSERT DATA {{ <{V}note{serial[0]}> <{V}text> "Butch {serial[0]}" }}'
            )
            db.execute_sparql(is4(i))

    results["write_then_read"] = {"ms": median_ms(write_then_read) / len(comments)}

    a, d = results["object_join_a_first"]["ms"], results["object_join_d_first"]["ms"]
    if max(a, d) / max(min(a, d), 1e-6) > JOIN_ORDER_FACTOR:
        failures.append(
            f"object join: the written orders differ {max(a, d) / max(min(a, d), 1e-6):.1f}x "
            f"(limit {JOIN_ORDER_FACTOR}x)"
        )
    ratio = results["values_1024"]["ms"] / max(results["values_1025"]["ms"], 1e-6)
    if ratio > CROSSOVER_FACTOR:
        failures.append(
            f"values_1024: {ratio:.2f}x values_1025 (limit {CROSSOVER_FACTOR}x)"
        )
    ratio = results["write_then_read"]["ms"] / max(results["is4"]["ms"], 1e-6)
    if ratio > WRITE_READ_FACTOR:
        failures.append(
            f"write_then_read: {ratio:.1f}x is4 (limit {WRITE_READ_FACTOR}x)"
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
    print(f"{'scenario':24} {'median ms':>10} {'baseline':>10} {'ratio':>7}")
    for name, r in results.items():
        b = base.get(name, {}).get("ms")
        ratio = f"{r['ms'] / b:6.2f}x" if b else ""
        print(f"{name:24} {r['ms']:10.3f} {b if b is not None else '':>10} {ratio:>7}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    for name in GUARD_RAILS:
        b = base.get(name, {}).get("ms")
        r = results.get(name)
        if b and r and r["ms"] > REGRESSION_FACTOR * b:
            failures.append(
                f"{name}: {r['ms'] / b:.2f}x the baseline (limit {REGRESSION_FACTOR}x)"
            )
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
