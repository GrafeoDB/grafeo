"""The Silicon Valley demo dataset.

An homage to the HBO series Silicon Valley; not affiliated with or endorsed by
HBO. All descriptions are original.
"""

from __future__ import annotations

import json
import re
import sys
from array import array
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Protocol

__all__ = [
    "VectorSet",
    "cypher",
    "edges",
    "gql",
    "load",
    "nodes",
    "timeline",
    "turtle",
    "vectors",
]

_DATA = "data/silicon_valley"
# The label and key of the node a statement inserts, as in `INSERT (:AI {key: 'anton', ...})`.
_FIRST_NODE = re.compile(r"^INSERT \(:(\w+) \{key: '([a-z0-9-]+)'")


class _Database(Protocol):
    """Anything with an `execute(str)` method, such as `grafeo.GrafeoDB`."""

    def execute(self, query: str, /) -> Any: ...


def _read(name: str) -> bytes:
    return files("grafeo_datasets").joinpath(_DATA, name).read_bytes()


def _text(name: str) -> str:
    return _read(name).decode("utf-8")


def nodes() -> list[dict[str, Any]]:
    """Nodes as dicts with `key`, `label`, `name`, `description` and `props`."""
    return json.loads(_text("graph.json"))["nodes"]


def edges() -> list[dict[str, Any]]:
    """Edges as dicts with `type`, `source` and `target` (node keys) and `props`."""
    return json.loads(_text("graph.json"))["edges"]


def timeline() -> list[dict[str, Any]]:
    """Dated facts with `subject`, `relation` and `object` (node keys).

    `from` and `to` are optional: `from` is absent when only the end is known,
    `to` is absent while the fact still holds. Dates are `YYYY-MM-DD`, or
    `YYYY-MM` when only the month is known.
    """
    return json.loads(_text("timeline.json"))


def gql() -> str:
    """The GQL script, one statement per line."""
    return _text("silicon-valley.gql")


def cypher() -> str:
    """The Cypher script, one statement per line."""
    return _text("silicon-valley.cypher")


def turtle() -> str:
    """The Turtle text. It carries no vectors."""
    return _text("silicon-valley.ttl")


@dataclass(frozen=True)
class VectorSet:
    """Description embeddings: `by_key` maps a node key to `dimension` unit-length floats."""

    model: str
    dimension: int
    by_key: dict[str, list[float]]


def vectors() -> VectorSet:
    """Embeddings of every node description, from the model named in `model`."""
    meta = json.loads(_text("vectors.json"))
    dimension: int = meta["dimension"]
    keys = meta["nodes"]
    data = array("f")
    data.frombytes(_read("vectors.f32"))
    if sys.byteorder == "big":
        data.byteswap()  # the file is little-endian
    if len(data) != len(keys) * dimension:
        raise ValueError(
            f"vectors.f32 holds {len(data)} floats, expected {len(keys)} x {dimension}."
        )
    by_key = {
        entry["key"]: data[entry["offset"] : entry["offset"] + dimension].tolist()
        for entry in keys
    }
    return VectorSet(model=meta["model"], dimension=dimension, by_key=by_key)


def _count(result: Any) -> int:
    return int(result.scalar())


def load(db: _Database, *, vectors: bool = False) -> int:
    """Run the GQL script (or the one with vectors) on `db`; return the statement count.

    Raises `ValueError` when the dataset is already in the database, fully or in
    part: the check looks for the node the script's first statement inserts, so
    a load that failed partway is refused too. Loading again after a partial
    failure needs an empty database.
    """
    script = _text("silicon-valley.vectors.gql") if vectors else gql()
    statements = [line for line in script.split("\n") if line.strip()]
    first = _FIRST_NODE.match(statements[0])
    if first is None:
        raise ValueError("The script's first statement does not insert a node.")
    label, key = first.groups()
    found = db.execute(f"MATCH (n:{label} {{key: '{key}'}}) RETURN count(n) AS c")
    if _count(found) > 0:
        raise ValueError(
            "The silicon-valley dataset is already in this database, fully or in part "
            f"(a node with key '{key}' exists)."
        )
    for statement in statements:
        db.execute(statement)
    return len(statements)
