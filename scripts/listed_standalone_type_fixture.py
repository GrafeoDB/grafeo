"""Writes the fixture of a standalone type a graph type took over, with the 0.6.2 wheel, which let it happen.

Run with Python holding the released grafeo 0.6.2, from the repository root:

    python scripts/listed_standalone_type_fixture.py

0.6.2 let a graph type that lists a standalone node type take it over: the type stopped applying in
the default graph, and two writes of the wrong type went in there (#764). 0.6.3 keeps a listed
standalone type in force, so this file is how the taken-over state stays reachable in a test: an
old file keeps its meaning, and `validate()` names the rows that would break the type.
"""

from pathlib import Path

import grafeo

if grafeo.__version__ != "0.6.2":
    raise RuntimeError(f"run with the 0.6.2 wheel, not {grafeo.__version__}")
OUT = Path("crates/grafeo-engine/tests/fixtures/listed_standalone_type/0.6.2.grafeo")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.unlink(missing_ok=True)

db = grafeo.GrafeoDB.open(str(OUT))
for statement in [
    # The reporter's steps (#764).
    "CREATE NODE TYPE A (h FLOAT64)",
    "INSERT (:A {k: 1, h: 1.5e0})",
    "CREATE GRAPH TYPE t (NODE TYPE A)",
    "MATCH (n:A {k: 1}) SET n.h = 'y'",
    "INSERT (:A {k: 3, h: 'z'})",
    # A row of the right type, which breaks nothing.
    "INSERT (:A {k: 4, h: 2.5e0})",
]:
    db.execute(statement)
db.close()
print(OUT, OUT.stat().st_size)
