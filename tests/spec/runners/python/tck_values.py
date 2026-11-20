"""The openCypher TCK's value notation: parse a cell and compare it to a result value.

The notation (tck/README.adoc, "Format of the expected results"): integers, floats
(`1.0`, `1e-3`, `NaN`, `Inf`, `-Inf`), single-quoted strings, `true`, `false`,
`null`, lists `[a, b]`, maps `{k: v}`, nodes `(:A:B {p: 1})`, relationships
`[:T {p: 1}]` and paths `<(:A)-[:T]->(:B)<-[:U]-()>`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Float:
    """A float cell, kept apart from integers (in Python `1 == 1.0`)."""

    value: float


@dataclass
class Node:
    labels: frozenset[str]
    properties: dict = field(default_factory=dict)


@dataclass
class Relationship:
    type: str
    properties: dict = field(default_factory=dict)


@dataclass
class Path:
    """Nodes and relationships in order; `forward[i]` is True when relationship i
    starts at node i (`-[]->`) and False when it starts at node i + 1 (`<-[]-`)."""

    nodes: list[Node]
    relationships: list[Relationship]
    forward: list[bool]


ESCAPES = {
    "'": "'",
    '"': '"',
    "\\": "\\",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


class Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def fail(self, what: str) -> ValueError:
        return ValueError(f"{what} at {self.at} in {self.text!r}")

    def skip(self) -> None:
        while self.at < len(self.text) and self.text[self.at].isspace():
            self.at += 1

    def peek(self, token: str) -> bool:
        self.skip()
        return self.text.startswith(token, self.at)

    def take(self, token: str) -> None:
        if not self.peek(token):
            raise self.fail(f"expected {token!r}")
        self.at += len(token)

    def name(self) -> str:
        self.skip()
        if self.peek("`"):
            end = self.text.index("`", self.at + 1)
            name, self.at = self.text[self.at + 1 : end], end + 1
            return name
        start = self.at
        while self.at < len(self.text) and (
            self.text[self.at].isalnum() or self.text[self.at] == "_"
        ):
            self.at += 1
        if start == self.at:
            raise self.fail("expected a name")
        return self.text[start : self.at]

    def value(self):
        self.skip()
        rest = self.text[self.at :]
        for word, value in (("null", None), ("true", True), ("false", False)):
            if rest.startswith(word) and not rest[len(word) : len(word) + 1].isalnum():
                self.at += len(word)
                return value
        for word, value in (("NaN", math.nan), ("-Inf", -math.inf), ("Inf", math.inf)):
            if rest.startswith(word):
                self.at += len(word)
                return Float(value)
        char = rest[:1]
        if char == "'":
            return self.string()
        if char == "{":
            return self.map()
        if char == "(":
            return self.node()
        if char == "<":
            return self.path()
        if rest.startswith("[:"):
            return self.relationship()
        if char == "[":
            return self.list()
        return self.number()

    def string(self) -> str:
        self.take("'")
        out = []
        while True:
            if self.at >= len(self.text):
                raise self.fail("unterminated string")
            char = self.text[self.at]
            if char == "'":
                self.at += 1
                return "".join(out)
            if char == "\\":
                code = self.text[self.at + 1]
                if code == "u":
                    out.append(chr(int(self.text[self.at + 2 : self.at + 6], 16)))
                    self.at += 6
                    continue
                out.append(ESCAPES.get(code, "\\" + code))
                self.at += 2
                continue
            out.append(char)
            self.at += 1

    def number(self):
        start = self.at
        while self.at < len(self.text) and self.text[self.at] in "+-0123456789.eE":
            self.at += 1
        text = self.text[start : self.at]
        if not text:
            raise self.fail("expected a value")
        if any(c in text for c in ".eE"):
            return Float(float(text))
        return int(text)

    def items(self, close: str, item):
        out = []
        if self.peek(close):
            self.take(close)
            return out
        while True:
            out.append(item())
            if self.peek(","):
                self.take(",")
                continue
            self.take(close)
            return out

    def list(self) -> list:
        self.take("[")
        return self.items("]", self.value)

    def map(self) -> dict:
        self.take("{")

        def entry():
            key = self.name()
            self.take(":")
            return key, self.value()

        return dict(self.items("}", entry))

    def node(self) -> Node:
        self.take("(")
        labels = []
        while self.peek(":"):
            self.take(":")
            labels.append(self.name())
        properties = self.map() if self.peek("{") else {}
        self.take(")")
        return Node(frozenset(labels), properties)

    def relationship(self) -> Relationship:
        self.take("[")
        self.take(":")
        rel_type = self.name()
        properties = self.map() if self.peek("{") else {}
        self.take("]")
        return Relationship(rel_type, properties)

    def path(self) -> Path:
        self.take("<")
        nodes, relationships, forward = [self.node()], [], []
        while not self.peek(">"):
            if self.peek("<-"):
                self.take("<-")
                relationships.append(self.relationship())
                self.take("-")
                forward.append(False)
            else:
                self.take("-")
                relationships.append(self.relationship())
                self.take("->")
                forward.append(True)
            nodes.append(self.node())
        self.take(">")
        return Path(nodes, relationships, forward)


def parse(text: str):
    """The value a TCK cell writes; raises ValueError on anything else."""
    parser = Parser(text)
    value = parser.value()
    parser.skip()
    if parser.at != len(text):
        raise parser.fail("unexpected text")
    return value


def _kind(actual) -> str:
    """The kind of a value in the neutral form of `values_cases.json`."""
    if actual is None:
        return "null"
    if isinstance(actual, bool):
        return "bool"
    if isinstance(actual, list):
        return "list"
    (key,) = actual
    return key


def _float(text: str) -> float:
    return {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}.get(text) or float(
        text
    )


def _same_float(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def _properties(expected: dict, actual: dict, flags: dict) -> bool:
    return expected.keys() == actual.keys() and all(
        matches(v, actual[k], **flags) for k, v in expected.items()
    )


def _node(expected: Node, actual: dict, flags: dict) -> bool:
    return expected.labels == frozenset(actual["labels"]) and _properties(
        expected.properties, actual["properties"], flags
    )


def _relationship(expected: Relationship, actual: dict, flags: dict) -> bool:
    return expected.type == actual["type"] and _properties(
        expected.properties, actual["properties"], flags
    )


def _path(expected: Path, actual: dict, flags: dict) -> bool:
    nodes, relationships = actual["nodes"], actual["relationships"]
    if len(nodes) != len(expected.nodes) or len(relationships) != len(
        expected.relationships
    ):
        return False
    for i, rel in enumerate(relationships):
        start, end = (
            (nodes[i], nodes[i + 1])
            if expected.forward[i]
            else (nodes[i + 1], nodes[i])
        )
        if (rel["source"], rel["target"]) != (start["id"], end["id"]):
            return False
        if not _relationship(expected.relationships[i], rel, flags):
            return False
    return all(_node(e, a, flags) for e, a in zip(expected.nodes, nodes, strict=True))


def _bag(expected: list, actual: list, flags: dict) -> bool:
    if len(expected) != len(actual):
        return False
    left = list(actual)
    for item in expected:
        hit = next((i for i, a in enumerate(left) if matches(item, a, **flags)), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def matches(expected, actual, *, unordered_lists: bool = False) -> bool:
    """Whether a parsed TCK cell matches a result value in the neutral form, type-exact.

    The TCK's dialect, a string cell matching a temporal, is the caller's: it writes
    temporals as their Cypher ``toString()`` first, as every runner does.
    """
    flags = {"unordered_lists": unordered_lists}
    kind = _kind(actual)
    if expected is None or isinstance(expected, bool):
        return kind in ("null", "bool") and actual is expected
    if isinstance(expected, int):
        return kind == "int" and int(actual["int"]) == expected
    if isinstance(expected, Float):
        return kind == "float" and _same_float(expected.value, _float(actual["float"]))
    if isinstance(expected, str):
        return kind == "string" and actual["string"] == expected
    if isinstance(expected, list):
        if kind != "list":
            return False
        if unordered_lists:
            return _bag(expected, actual, flags)
        return len(expected) == len(actual) and all(
            matches(e, a, **flags) for e, a in zip(expected, actual, strict=True)
        )
    if isinstance(expected, dict):
        return kind == "map" and _properties(expected, actual["map"], flags)
    if isinstance(expected, Node):
        return kind == "node" and _node(expected, actual["node"], flags)
    if isinstance(expected, Relationship):
        return kind == "relationship" and _relationship(
            expected, actual["relationship"], flags
        )
    if isinstance(expected, Path):
        return kind == "path" and _path(expected, actual["path"], flags)
    raise TypeError(f"not a TCK value: {expected!r}")


def _entity(value: dict, reserved: tuple[str, ...], ty) -> dict:
    properties = {
        k: to_neutral(v, _member(ty, k)) for k, v in value.items() if k not in reserved
    }
    return {"id": value["_id"], "properties": properties}


def _member(ty, key):
    """The type of a map's value at ``key``: by key in a struct, the one type of a map."""
    if isinstance(ty, dict) and "struct" in ty:
        return ty["struct"].get(key, "any")
    if isinstance(ty, dict) and "map" in ty:
        return ty["map"]
    return "any"


def _item(ty, index: int):
    """The type of a list's item at ``index``: the one type of a list, by position in a tuple."""
    if isinstance(ty, dict) and "list" in ty:
        return ty["list"]
    if isinstance(ty, dict) and "tuple" in ty:
        items = ty["tuple"]
        return items[index] if index < len(items) else "any"
    return "any"


def _node_of(value: dict) -> dict:
    node = _entity(value, ("_id", "_labels"), "any")
    return {**node, "labels": list(value["_labels"])}


def _relationship_of(value: dict) -> dict:
    rel = _entity(value, ("_id", "_type", "_source", "_target"), "any")
    ends = {"source": value["_source"], "target": value["_target"]}
    return {**rel, "type": value["_type"], **ends}


def to_neutral(value, ty="any"):
    """A result value as the binding gives it, in the neutral form ``matches`` compares.

    ``ty`` is the value's cell type (``cell_type(row, column)``): the one place that
    recognises entities. A dict is a node where its type is ``"node"``, a relationship
    where it is ``"edge"`` and a path where it is ``"path"``; anywhere else it is a map,
    whatever its keys. Lists and maps hand their items the types inside theirs.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "NaN"}
        if math.isinf(value):
            return {"float": "Inf" if value > 0 else "-Inf"}
        return {"float": repr(value)}
    if isinstance(value, str):
        return {"string": value}
    if isinstance(value, (list, tuple)):
        return [to_neutral(v, _item(ty, i)) for i, v in enumerate(value)]
    if isinstance(value, dict):
        if ty == "node":
            return {"node": _node_of(value)}
        if ty == "edge":
            return {"relationship": _relationship_of(value)}
        if ty == "path":
            return {
                "path": {
                    "nodes": [_node_of(n) for n in value["nodes"]],
                    "relationships": [_relationship_of(e) for e in value["edges"]],
                }
            }
        return {
            "map": {str(k): to_neutral(v, _member(ty, k)) for k, v in value.items()}
        }
    # A value no cell writes (a temporal outside the TCK dialect): it matches no cell
    return {"other": repr(value)}


def table_rows(table: str) -> list[list[str]]:
    """The rows of a ``values:`` table, each a list of cell texts (a backslash before a
    pipe makes the pipe part of the cell)."""
    rows = []
    for line in table.splitlines():
        line = line.strip()
        if not line:
            continue
        inner = line[1:-1] if line.startswith("|") and line.endswith("|") else line
        cells, cell, index = [], "", 0
        while index < len(inner):
            char = inner[index]
            if char == "\\" and inner[index + 1 : index + 2] == "|":
                cell += "|"
                index += 2
                continue
            if char == "|":
                cells.append(cell.strip())
                cell = ""
            else:
                cell += char
            index += 1
        cells.append(cell.strip())
        rows.append(cells)
    return rows
