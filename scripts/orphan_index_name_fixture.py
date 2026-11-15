"""Writes the fixture of an index name that no graph has an index for, with the 0.6.1 wheel, whose
`DROP GRAPH` kept the names of the indexes it dropped.

Run with Python holding the released grafeo 0.6.1, from the repository root:

    uv run --no-project --python 3.13 --with grafeo==0.6.1 python scripts/orphan_index_name_fixture.py

The default graph has an index named `kept` on `File.id` (0.6.1 built it on `id` for every
label). The graph `gone` had an index named `orphan` on `Doc.code`, a property no index of the
default graph is on, and was dropped: 0.6.1 kept the name `orphan` in the catalog, with no graph
holding its index, so `CREATE INDEX orphan` of another definition failed and `SHOW INDEXES` listed
it in every graph. Later versions drop such a name with its last index, so this file is how an
orphan name stays reachable in a test.
"""

import sys
from pathlib import Path

import grafeo

assert grafeo.__version__ == "0.6.1", grafeo.__version__
OUT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "crates/grafeo-engine/tests/fixtures/orphan_index_name/0.6.1.grafeo"
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.unlink(missing_ok=True)

db = grafeo.GrafeoDB.open(str(OUT))

for statement in [
    "CREATE INDEX kept FOR (n:File) ON (n.id)",
    "INSERT (:File {id: 1})",
    "CREATE GRAPH gone",
    "USE GRAPH gone",
    "CREATE INDEX orphan FOR (n:Doc) ON (n.code)",
    "INSERT (:Doc {code: 1})",
    "USE GRAPH default",
    "DROP GRAPH gone",
]:
    db.execute(statement)
names = sorted(
    row["name"] for row in db.execute("SHOW INDEXES") if row["name"] is not None
)
assert names == ["kept", "orphan"], names
db.close()
print(OUT, OUT.stat().st_size)
