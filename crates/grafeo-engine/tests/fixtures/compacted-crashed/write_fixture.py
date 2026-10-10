"""Write `<version>/crashed.grafeo`: writes after `compact()`, then an exit without `close()`.

Up to 0.5.44, a direct call after `compact()` (`set_node_property`, `delete_node`, ...)
wrote its own WAL record, and a query after it wrote none (#558). This writes a few
people and edges, compacts them into the base and checkpoints it into the file, changes
base nodes and edges with direct calls, runs one query, and exits without `close()`, so
the sidecar WAL (`crashed.grafeo.wal/`) holds the direct calls. `tests/compacted_file_in_every_build.rs`
opens the file; its tests describe what it holds. Run this with the released wheel, from
the repository root:

    uv run --no-project --python 3.13 --with grafeo==0.5.44 python \\
        crates/grafeo-engine/tests/fixtures/compacted-crashed/write_fixture.py

Regenerate it only to add content.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import grafeo

OUT = Path("crates/grafeo-engine/tests/fixtures/compacted-crashed")

PEOPLE = [
    ("Alix", "Amsterdam"),
    ("Gus", "Berlin"),
    ("Vincent", "Paris"),
    ("Mia", "Prague"),
    ("Butch", "Barcelona"),
]
KNOWS = [("Alix", "Gus"), ("Gus", "Vincent"), ("Vincent", "Mia")]


def main() -> None:
    if len(sys.argv) != 1:
        print("usage: write_fixture.py", file=sys.stderr)
        sys.exit(2)
    out = OUT / grafeo.__version__
    target = out / "crashed.grafeo"
    if target.exists():
        print(f"{target} exists; remove it first to regenerate it", file=sys.stderr)
        sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)
    db = grafeo.GrafeoDB(str(target))

    # The compacted base.
    people = {
        name: db.create_node(["Person"], {"name": name, "city": city}).id
        for name, city in PEOPLE
    }
    knows = {
        (source, target): db.create_edge(
            people[source], people[target], "KNOWS", {"since": 2019}
        ).id
        for source, target in KNOWS
    }
    db.compact()
    # `compact()` builds the base in memory; the checkpoint writes it to the file and
    # empties the WAL.
    db.wal_checkpoint()

    # Direct calls on nodes and edges of the base, and a new node with an edge to one:
    # each writes its WAL record.
    db.set_node_property(people["Alix"], "city", "Berlin")
    db.add_node_label(people["Gus"], "Employee")
    db.remove_node_property(people["Vincent"], "city")
    db.delete_edge(knows[("Gus", "Vincent")])
    db.delete_node(people["Butch"])
    jules = db.create_node(["Person"], {"name": "Jules", "city": "Amsterdam"}).id
    db.create_edge(jules, people["Mia"], "KNOWS", {"since": 2088})

    # A query: 0.5.44 wrote no WAL record for it (#558).
    db.execute("INSERT (:Person {name: 'Django', city: 'Paris'})")

    # Leave the changes in the WAL: no close, no checkpoint.
    for spill in out.glob("crashed.grafeo.spill"):
        shutil.rmtree(spill, ignore_errors=True)
    print(target, target.stat().st_size)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
