"""Write the databases that `crates/grafeo-engine/tests/released_formats.rs` opens.

A released Grafeo writes one database in three layouts, so the tests can check that
today's code opens every format that has shipped:

- `closed.grafeo`: a single file, closed cleanly, so everything is in the file.
- `unflushed.grafeo` and `unflushed.grafeo.wal/`: a single file whose sidecar WAL holds the
  second half of the changes, because the process exited without `close()`.
- `directory/`: a WAL-directory database, closed cleanly.

Each layout gets the same changes in two sessions: `first_half`, a reopen, then
`second_half`. Run it with the release from PyPI, from the repository root:

    uv run --no-project --python 3.13 --with grafeo==0.5.44 python scripts/released_fixtures.py

The files go to `crates/grafeo-engine/tests/fixtures/released/<version>/`. Regenerate a
version only to add content; the tests describe what each file holds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import grafeo

OUT = Path("crates/grafeo-engine/tests/fixtures/released")

FIRST_HALF = [
    # Schema: a default value, a parent type, endpoint restrictions, a graph type,
    # a named and an unnamed constraint, and a property index.
    "CREATE NODE TYPE City (name STRING NOT NULL, country STRING DEFAULT 'NL')",
    "CREATE NODE TYPE Capital EXTENDS City (since INT64)",
    "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64 DEFAULT 88)",
    "CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)",
    "CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE",
    "CREATE CONSTRAINT FOR (p:Person) ON (p.name) NOT NULL",
    "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    # Every value type on one node, a node with two labels, and one to delete later.
    (
        "INSERT (:Person {name: 'Alix', email: 'alix@example.org', age: 30, score: -3,"
        " height: 1.88, active: true, tags: ['amsterdam', 'jazz'],"
        " address: {city: 'Amsterdam', number: 19},"
        " born: date('1994-03-19'), seen: zoned_datetime('2024-03-19T08:30:00+01:00'),"
        " stay: duration('P3D'), embedding: vector([3.0, 19.0, 88.0])})"
    ),
    "INSERT (:Person:Employee {name: 'Gus', email: 'gus@example.org', age: 25})",
    "INSERT (:Person {name: 'Vincent', email: 'vincent@example.org'})",
    "INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin', country: 'DE'})",
    "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS {since: 2019}]->(g)",
    "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Amsterdam'}) INSERT (a)-[:LIVES_IN]->(c)",
    # Documents for a vector and a text index.
    (
        "INSERT (:Document {title: 'Canals', content: 'boats on the canals of Amsterdam',"
        " embedding: vector([1.0, 0.0, 0.0])}),"
        " (:Document {title: 'Museums', content: 'paintings in the museums of Berlin',"
        " embedding: vector([0.0, 1.0, 0.0])})"
    ),
    "CREATE VECTOR INDEX doc_embedding ON :Document(embedding)",
    "CREATE INDEX doc_content FOR (d:Document) ON (d.content) USING TEXT",
    # A named graph bound to the graph type.
    "CREATE GRAPH trips TYPED travel",
]

FIRST_HALF_IN_TRIPS = [
    "INSERT (:City {name: 'Paris', country: 'FR'}), (:City {name: 'Prague', country: 'CZ'})",
    "MATCH (p:City {name: 'Paris'}), (q:City {name: 'Prague'}) INSERT (p)-[:ROUTE {km: 1030}]->(q)",
]

FIRST_HALF_SPARQL = [
    (
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> ."
        ' <http://example.org/alix> <http://example.org/name> "Alix" . }'
    ),
]

SECOND_HALF = [
    # Schema in the WAL: the same kinds of definitions as the first half.
    "CREATE NODE TYPE Museum (name STRING NOT NULL, open BOOL DEFAULT true)",
    "CREATE EDGE TYPE IN_CITY CONNECTING (Museum) TO (City)",
    "CREATE GRAPH TYPE culture (NODE TYPE Museum, NODE TYPE City, EDGE TYPE IN_CITY)",
    "CREATE CONSTRAINT museum_name FOR (m:Museum) ON (m.name) UNIQUE",
    "CREATE INDEX museum_name_index FOR (m:Museum) ON (m.name)",
    # Changes to the first half's data: an update, a removed property, a new label,
    # a deleted node with its edges.
    "MATCH (a:Person {name: 'Alix'}) SET a.age = 31 REMOVE a.score",
    "MATCH (g:Person {name: 'Gus'}) SET g:Manager",
    "MATCH (v:Person {name: 'Vincent'}) DETACH DELETE v",
    "INSERT (:Person {name: 'Mia', email: 'mia@example.org', embedding: vector([0.0, 0.0, 1.0])})",
    "INSERT (:Museum {name: 'Rijksmuseum'})",
    "MATCH (m:Museum {name: 'Rijksmuseum'}), (c:City {name: 'Amsterdam'}) INSERT (m)-[:IN_CITY]->(c)",
    (
        "INSERT (:Document {title: 'Bridges', content: 'bridges over the canals of Prague',"
        " embedding: vector([0.0, 0.0, 1.0])})"
    ),
    # A second named graph, created in the WAL.
    "CREATE GRAPH museums TYPED culture",
]

SECOND_HALF_IN_MUSEUMS = [
    "INSERT (:Museum {name: 'Louvre'})",
]

SECOND_HALF_SPARQL = [
    "INSERT DATA { <http://example.org/gus> <http://example.org/knows> <http://example.org/mia> . }",
    'DELETE DATA { <http://example.org/alix> <http://example.org/name> "Alix" . }',
]


def run(db: grafeo.GrafeoDB, statements: list[str]) -> None:
    for statement in statements:
        db.execute(statement)


def run_sparql(db: grafeo.GrafeoDB, statements: list[str]) -> None:
    for statement in statements:
        db.execute_sparql(statement)


def first_half(db: grafeo.GrafeoDB) -> None:
    run(db, FIRST_HALF)
    run_sparql(db, FIRST_HALF_SPARQL)
    db.set_graph("trips")
    run(db, FIRST_HALF_IN_TRIPS)
    db.reset_graph()


def second_half(db: grafeo.GrafeoDB) -> None:
    run(db, SECOND_HALF)
    run_sparql(db, SECOND_HALF_SPARQL)
    db.set_graph("museums")
    run(db, SECOND_HALF_IN_MUSEUMS)
    db.reset_graph()


def write(path: Path, *, close_at_end: bool) -> None:
    db = grafeo.GrafeoDB(str(path))
    first_half(db)
    db.close()
    db = grafeo.GrafeoDB(str(path))
    second_half(db)
    if close_at_end:
        db.close()
    else:
        # Leave the second half in the WAL: no close, no checkpoint.
        sys.stdout.flush()
        os._exit(0)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--unflushed":
        write(Path(sys.argv[2]), close_at_end=False)
        return
    out = OUT / grafeo.__version__
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write(out / "closed.grafeo", close_at_end=True)
    write(out / "directory", close_at_end=True)
    subprocess.run(
        [sys.executable, __file__, "--unflushed", str(out / "unflushed.grafeo")],
        check=True,
    )
    # Opening creates an empty spill directory next to each database.
    for spill in out.glob("*.spill"):
        spill.rmdir()
    for path in sorted(out.rglob("*")):
        size = path.stat().st_size if path.is_file() else ""
        print(path.relative_to(out), size)


if __name__ == "__main__":
    main()
