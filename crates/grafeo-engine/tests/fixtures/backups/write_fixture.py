"""Write `<version>/`: a backup chain (a full backup and two incremental ones).

Up to 0.5.44 the backup manifest (`backup_manifest.json`) held bincode; 0.6 writes
JSON and still reads the bincode manifest. The manifest is kept as
`backup_manifest.bincode` (0.5.x), so text hooks leave its bytes alone (0.6's JSON
manifest keeps its name, with a final newline); `tests/backup_restore.rs`
copies it back under its real name and restores the chain; its tests describe what
it holds. Run this with the released wheel, from the
repository root:

    uv run --no-project --python 3.13 --with grafeo==0.5.44 python \\
        crates/grafeo-engine/tests/fixtures/backups/write_fixture.py

Regenerate it only to add content.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import grafeo

OUT = Path("crates/grafeo-engine/tests/fixtures/backups")

# Each step runs its statements, then takes its backup into the chain.
STEPS = [
    (
        "full",
        [
            "INSERT (:Person {name: 'Alix', city: 'Amsterdam'})",
            "INSERT (:Person {name: 'Gus', city: 'Berlin'})",
            "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS {since: 2019}]->(g)",
        ],
    ),
    ("incremental", ["INSERT (:Person {name: 'Vincent', city: 'Paris'})"]),
    (
        "incremental",
        [
            "INSERT (:Person {name: 'Mia', city: 'Prague'})",
            "MATCH (m:Person {name: 'Mia'}), (a:Person {name: 'Alix'}) INSERT (m)-[:KNOWS {since: 1988}]->(a)",
            "MATCH (g:Person {name: 'Gus'}) SET g.city = 'Barcelona'",
        ],
    ),
]


def main() -> None:
    if len(sys.argv) != 1:
        print("usage: write_fixture.py", file=sys.stderr)
        sys.exit(2)
    out = OUT / grafeo.__version__
    if out.exists():
        print(f"{out} exists; remove it first to regenerate it", file=sys.stderr)
        sys.exit(2)
    with tempfile.TemporaryDirectory() as scratch:
        backups = Path(scratch) / "backups"
        db = grafeo.GrafeoDB(str(Path(scratch) / "source.grafeo"))
        epochs = []
        for kind, statements in STEPS:
            for statement in statements:
                db.execute(statement)
            if kind == "full":
                db.backup_full(str(backups))
            else:
                db.backup_incremental(str(backups))
            epochs.append(db.current_epoch())
        db.close()
        shutil.copytree(backups, out)
    version = tuple(int(part) for part in grafeo.__version__.split(".")[:2])
    writes_json = version >= (0, 6)
    manifest = out / "backup_manifest.json"
    if writes_json:
        # JSON text: given the final newline the end-of-file hook would add, which JSON
        # ignores, so the committed bytes are the ones written here.
        text = manifest.read_bytes()
        if not text.endswith(b"\n"):
            manifest.write_bytes(text + b"\n")
    else:
        # Bincode in a `.json` file: kept under a name text hooks leave alone.
        manifest.rename(out / "backup_manifest.bincode")
    for path in sorted(out.iterdir()):
        print(path, path.stat().st_size)
    print("epochs after each backup:", epochs)


if __name__ == "__main__":
    main()
