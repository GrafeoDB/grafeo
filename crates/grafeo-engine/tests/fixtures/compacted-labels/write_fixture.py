"""Write `<version>/labels.grafeo`: nodes with several labels through `compact()` (#595).

Up to 0.5.44, `compact()` stored a node with several labels under one label, its
labels joined with `|`, and a write after it that matched the node without a label
copied it to the overlay under that one label. `tests/compacted_labels.rs` opens the
file; its tests describe what it holds. Run this with the released wheel, from the
repository root:

    uv run --no-project --python 3.13 --with grafeo==0.5.44 python \\
        crates/grafeo-engine/tests/fixtures/compacted-labels/write_fixture.py

Regenerate it only to add content.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import grafeo

OUT = Path("crates/grafeo-engine/tests/fixtures/compacted-labels")

# The compacted base: the nodes of the issue, a second node with two labels that
# nothing changes later, one with three labels, one to delete later, and an edge
# between two nodes with several labels.
BEFORE_COMPACT = [
    "INSERT (:Graph:Repository {name: 'Alix'}), (:Graph {name: 'Gus'})",
    "INSERT (:Graph:Repository {name: 'Vincent'}), (:Archive:Graph:Repository {name: 'Mia'})",
    "INSERT (:Graph:Repository {name: 'Butch'})",
    "MATCH (a {name: 'Alix'}), (v {name: 'Vincent'}) INSERT (a)-[:FORKED_FROM {since: 2019}]->(v)",
]

# The overlay: a property and a label set on Alix, a new node with two labels, an
# edge that ends at Mia, and Butch deleted. Each matches without a label: after
# `compact()`, 0.5.44 found none of these nodes by their labels.
AFTER_COMPACT = [
    "MATCH (n {name: 'Alix'}) SET n.city = 'Amsterdam'",
    "MATCH (n {name: 'Alix'}) SET n:Starred",
    "INSERT (:Graph:Repository {name: 'Jules'})",
    "MATCH (j {name: 'Jules'}), (m {name: 'Mia'}) INSERT (j)-[:FORKED_FROM {since: 2088}]->(m)",
    "MATCH (n {name: 'Butch'}) DETACH DELETE n",
]


def main() -> None:
    if len(sys.argv) != 1:
        print("usage: write_fixture.py", file=sys.stderr)
        sys.exit(2)
    out = OUT / grafeo.__version__
    target = out / "labels.grafeo"
    if target.exists():
        print(f"{target} exists; remove it first to regenerate it", file=sys.stderr)
        sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)
    db = grafeo.GrafeoDB(str(target))
    for statement in BEFORE_COMPACT:
        db.execute(statement)
    db.compact()
    for statement in AFTER_COMPACT:
        db.execute(statement)
    db.close()
    for spill in out.glob("labels.grafeo.spill"):
        shutil.rmtree(spill)
    print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
