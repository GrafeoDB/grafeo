"""Writes the fixture of a schema's closed graph type that `ALTER GRAPH TYPE ... ADD` grew, with the 0.6.1
wheel, which stored the added types under their bare names.

Run with Python holding the released grafeo 0.6.1, from the repository root:

    python scripts/schema_alter_fixture.py

Under `SESSION SET SCHEMA s`, `CREATE GRAPH TYPE held (NODE TYPE Held, ...)` lists `s/Held`, while
0.6.1's `ALTER GRAPH TYPE held ADD NODE TYPE Extra` listed `Extra` alone. Later versions list both
qualified, so this file is how a bare entry stays reachable in a test.
"""

import sys
from pathlib import Path

import grafeo

assert grafeo.__version__ == "0.6.1", grafeo.__version__
OUT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "crates/grafeo-engine/tests/fixtures/schema_alter_bare/0.6.1.grafeo"
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.unlink(missing_ok=True)

db = grafeo.GrafeoDB.open(str(OUT))

for statement in [
    "CREATE SCHEMA s",
    "SESSION SET SCHEMA s",
    "CREATE NODE TYPE Held (name STRING)",
    "CREATE NODE TYPE Extra (k INTEGER)",
    "CREATE EDGE TYPE Link (w INTEGER)",
    "CREATE EDGE TYPE Added (w INTEGER)",
    "CREATE GRAPH TYPE held (NODE TYPE Held, EDGE TYPE Link)",
    "CREATE GRAPH g TYPED held",
    "ALTER GRAPH TYPE held ADD NODE TYPE Extra",
    "ALTER GRAPH TYPE held ADD EDGE TYPE Added",
    "USE GRAPH g",
    "INSERT (:Held {name: 'Alix'})-[:Link {w: 1}]->(:Extra {k: 1})-[:Added {w: 2}]->(:Held {name: 'Gus'})",
]:
    db.execute(statement)
db.close()
print(OUT, OUT.stat().st_size)
