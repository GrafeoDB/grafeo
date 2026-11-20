"""Parse the Gherkin subset the openCypher TCK uses.

Reads a `.feature` file into its scenarios, each a list of steps with their
docstring or table. A `Scenario Outline` becomes one scenario per row of its
`Examples`, with `<name>` replaced in the step texts, docstrings and tables. A
`Background` is put in front of every scenario's steps. Tags (`@ignore`) are
kept on the scenario they precede.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STEP = re.compile(r"(Given|When|Then|And|But)\s+(.*)")


@dataclass
class Step:
    """One step: its text without the keyword, and its docstring or table."""

    text: str
    docstring: str | None = None
    table: list[list[str]] | None = None


@dataclass
class Scenario:
    """One scenario, an outline already expanded to one example."""

    name: str
    tags: list[str]
    steps: list[Step]


@dataclass
class Feature:
    name: str
    scenarios: list[Scenario] = field(default_factory=list)


def table_row(line: str) -> list[str]:
    """The cells of a `| a | b |` row, Gherkin's `\\|` and `\\\\` undone.

    `\\n` is left as written: the TCK's value parser reads it as a string escape.
    """
    cells, cell = [], ""
    for token in re.findall(r"\\[\\|]|.", line.strip()[1:-1]):
        if token == "|":
            cells.append(cell.strip())
            cell = ""
        else:
            cell += token[-1] if token in ("\\\\", "\\|") else token
    return [*cells, cell.strip()]


def substitute(text: str, values: dict[str, str]) -> str:
    return re.sub(r"<([^<>]+)>", lambda m: values.get(m.group(1), m.group(0)), text)


def expand(step: Step, values: dict[str, str]) -> Step:
    return Step(
        substitute(step.text, values),
        None if step.docstring is None else substitute(step.docstring, values),
        None
        if step.table is None
        else [[substitute(c, values) for c in row] for row in step.table],
    )


@dataclass
class _Open:
    """The scenario being read: an outline collects its examples until the next one."""

    name: str
    tags: list[str]
    outline: bool
    steps: list[Step] = field(default_factory=list)
    header: list[str] | None = None
    examples: list[list[str]] = field(default_factory=list)

    def scenarios(self, background: list[Step]) -> list[Scenario]:
        steps = background + self.steps
        if not self.outline:
            return [Scenario(self.name, self.tags, steps)]
        out = []
        for number, row in enumerate(self.examples, 1):
            values = dict(zip(self.header or [], row, strict=True))
            name = f"{substitute(self.name, values)} #{number}"
            out.append(
                Scenario(name, self.tags, [expand(step, values) for step in steps])
            )
        return out


def parse_feature(text: str) -> Feature:
    """The feature's scenarios in file order, outlines expanded."""
    lines = text.splitlines()
    feature = Feature("")
    background: list[Step] = []
    tags: list[str] = []
    current: _Open | None = None
    target: list[Step] | None = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            tags += line.split()
        elif line.startswith("Feature:"):
            feature.name = line.removeprefix("Feature:").strip()
        elif line.startswith("Background:"):
            target = background
        elif line.startswith(("Scenario:", "Scenario Outline:")):
            if current is not None:
                feature.scenarios += current.scenarios(background)
            outline = line.startswith("Scenario Outline:")
            current = _Open(line.split(":", 1)[1].strip(), tags, outline)
            tags = []
            target = current.steps
        elif line.startswith("Examples:"):
            target = None
        elif line.startswith('"""') and target:
            body = []
            while not lines[index].strip().startswith('"""'):
                body.append(lines[index])
                index += 1
            index += 1
            indent = min(
                (len(b) - len(b.lstrip()) for b in body if b.strip()), default=0
            )
            target[-1].docstring = "\n".join(b[indent:] for b in body).strip("\n")
        elif line.startswith("|") and target:
            target[-1].table = (target[-1].table or []) + [table_row(line)]
        elif line.startswith("|") and current is not None:
            if current.header is None:
                current.header = table_row(line)
            else:
                current.examples.append(table_row(line))
        else:
            match = STEP.match(line)
            if not match or target is None:
                raise ValueError(f"line {index}: not a step: {line}")
            target.append(Step(match.group(2)))
    if current is not None:
        feature.scenarios += current.scenarios(background)
    return feature
