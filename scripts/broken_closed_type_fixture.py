"""Writes the fixture of a closed graph that breaks its type, with the 0.6.1 wheel, which let it happen.

Run with Python holding the released grafeo 0.6.1, from the repository root:

    python scripts/broken_closed_type_fixture.py

0.6.1 refused none of these: a type change over held data, `REMOVE` of a last label, of a label a
held property needs and of a label an edge needs, and `DROP NODE TYPE` of a listed type. Later
versions refuse them, so this file is how the state stays reachable in a test.
"""

from pathlib import Path

import grafeo

if grafeo.__version__ != "0.6.1":
    raise RuntimeError(f"run with the 0.6.1 wheel, not {grafeo.__version__}")
OUT = Path("crates/grafeo-engine/tests/fixtures/closed_type_broken/0.6.1.grafeo")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.unlink(missing_ok=True)

db = grafeo.GrafeoDB.open(str(OUT))
for statement in [
    "CREATE NODE TYPE Far (k INTEGER, name STRING)",
    "CREATE NODE TYPE Near (name STRING)",
    "CREATE NODE TYPE Free (id STRING, k INTEGER, name STRING, extra INTEGER)",
    "CREATE NODE TYPE Gone (name STRING, anything INTEGER)",
    "CREATE EDGE TYPE R CONNECTING (Far) TO (Near)",
    "CREATE GRAPH TYPE atlas (NODE TYPE Far, NODE TYPE Near, NODE TYPE Free, NODE TYPE Gone, EDGE TYPE R)",
    "CREATE GRAPH g TYPED atlas",
    "USE GRAPH g",
    # A held value its type no longer declares.
    "INSERT (:Far:Free {name: 'Alix', extra: 1})",
    "ALTER NODE TYPE Free DROP PROPERTY extra",
    # A node with no label.
    "INSERT (:Free {name: 'Gus'})",
    "MATCH (x {name: 'Gus'}) REMOVE x:Free",
    # An edge whose source no longer fits its type.
    "INSERT (:Far:Free {name: 'Vincent'})-[:R]->(:Near {name: 'Jules'})",
    "MATCH (x {name: 'Vincent'}) REMOVE x:Far",
    # A node of a listed type whose definition was dropped.
    "INSERT (:Gone {name: 'Mia', anything: 7})",
    "DROP NODE TYPE Gone",
]:
    db.execute(statement)
db.close()
print(OUT, OUT.stat().st_size)
