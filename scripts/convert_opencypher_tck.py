#!/usr/bin/env python3
"""Convert the openCypher TCK's feature files to .gtest files.

Reads `tck/features` of an openCypher checkout and writes one .gtest file per
feature under `tests/spec/lpg/cypher/tck/`, mirroring its folders. Every scenario
becomes a case (an outline one per example); a control query becomes a second
case whose setup ends with the scenario's query. Each case carries what the TCK
expects:

- result rows as `values:`, the TCK's own table, compared type-exact
  (`values_dialect: tck` lets a quoted cell match a temporal by its Cypher
  `toString()`); "in order" adds `ordered`, "ignoring element order for lists"
  adds `unordered_lists`;
- side effects as `side_effects:`, "no side effects" and every error as `{}`;
- an error as the GQLSTATUS class of its TCK type (`gql_status: "42"`), with the
  TCK's type, phase and detail in a comment;
- the TCK's test procedures as GQL `CREATE PROCEDURE` statements in `setup_gql:`.

`tests/spec/lpg/cypher/tck/known_failures.toml` lists, by feature, the cases
Grafeo does not pass: an `issue` makes a strict `xfail:` (with `passes_in`, the
runners where the case passes all the same), a `reason` leaves the
case out (named in the file's header). A `gql_class` with a `note` expects GQL's
class where a General Rule of GQL gives another than the TCK's, and says so in a
comment. An entry that names no case stops the
conversion. A step the converter does not know stops it too.

Usage:
    python scripts/convert_opencypher_tck.py --tck <openCypher checkout>/tck
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tests/spec/runners/python")
)

import tck_values  # noqa: E402

import opencypher_gherkin as gherkin  # noqa: E402

TAG = "2024.3"
OUTPUT = Path("tests/spec/lpg/cypher/tck")

# The runners whose bindings do not hand temporals back without loss: they skip a
# case that builds or returns one. WASM returns a temporal as its text, which passed
# back is a string; typed temporals there come in 0.7.0
TEMPORAL_GAP = {
    "runners": ["wasm"],
    "issue": "fix in 0.7.0",
    "reason": "temporals come back from this binding as text, which passed back is a string",
}
TEMPORAL = re.compile(
    r"\b(?:date|datetime|localtime|localdatetime|time|duration)\s*(?:\(|\.\w)"
)

# The GQLSTATUS class of each TCK error type raised at run time. An error the TCK raises at
# compile time is one GQL's Syntax Rules catch (static types, grammar), class 42; where a
# General Rule of GQL decides instead, known_failures.toml gives its `gql_class`
ERROR_CLASSES = {
    "SyntaxError": "42",
    "SemanticError": "42",
    "ParameterMissing": "42",
    "ProcedureError": "42",
    "PropertyNotFound": "42",
    "TypeError": "22",
    "ArgumentError": "22",
    "EntityNotFound": "22",
    "ConstraintVerificationFailed": "G2",
    "ConstraintValidationFailed": "G2",
}
# Where the type is too coarse, the detail decides
DETAIL_CLASSES = {"DeleteConnectedNode": "G1"}
# Classes the first run checks against what Grafeo raises
PROVISIONAL = {"EntityNotFound"}

PROCEDURE_TYPES = {
    "INTEGER": "INT64",
    "FLOAT": "FLOAT64",
    "STRING": "STRING",
    "BOOLEAN": "BOOLEAN",
    "NUMBER": "ANY",
    "ANY": "ANY",
    "MAP": "ANY",
    "NODE": "ANY",
    "RELATIONSHIP": "ANY",
    "PATH": "ANY",
}

# String values of the TCK's data that name another product, by the SHA-256 of the quoted
# literal so this file does not spell them, with the neutral value that replaces them
NEUTRAL_VALUES = {
    "88f3d85fcd6a1171b449e3a3dbda4657c979938987d42668cc22f25ce60e42cb": "'Grafeo'",
}
NEUTRAL_NOTE = "a product name in the TCK's data is replaced by 'Grafeo'"

ERROR = re.compile(
    r"an? (\w+) should be raised at (compile time|runtime|any time): (\S+)"
)
PROCEDURE = re.compile(r"there exists a procedure ([\w.]+)\((.*?)\) :: \((.*?)\)\s*:?$")


@dataclass
class Case:
    name: str
    title: str
    setup: list[str] = field(default_factory=list)
    setup_gql: list[str] = field(default_factory=list)
    params: list[tuple[str, str]] = field(default_factory=list)
    query: str = ""
    expect: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    xfail: tuple[str, str, list[str]] | None = None
    runner_gap: dict | None = None


def slug(text: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return name if name and not name[0].isdigit() else f"s{name}"


def param_text(value) -> str:
    """A TCK value as a .gtest parameter: quoted for a string, JSON for a container."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json_text(value)


def json_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tck_values.Float):
        return repr(value.value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(json_text(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{json.dumps(k)}: {json_text(v)}" for k, v in value.items())
            + "}"
        )
    raise ValueError(f"no parameter form for {value!r}")


def gql_literal(value) -> str:
    """A TCK value as a GQL literal, for a procedure's table."""
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, list):
        return "[" + ", ".join(gql_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"`{k}`: {gql_literal(v)}" for k, v in value.items()) + "}"
        )
    return json_text(value)


def procedure_type(text: str) -> str:
    base = text.strip().rstrip("?").strip()
    if base.startswith("LIST"):
        return "ANY"
    if base not in PROCEDURE_TYPES:
        raise ValueError(f"no GQL type for procedure type {text!r}")
    return PROCEDURE_TYPES[base]


def signature(text: str) -> list[tuple[str, str]]:
    out = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, kind = part.partition("::")
        out.append((name.strip(), procedure_type(kind)))
    return out


def create_procedure(step: gherkin.Step) -> str:
    """A GQL `CREATE PROCEDURE` that answers from the TCK's table."""
    match = PROCEDURE.match(step.text)
    if not match:
        raise ValueError(f"procedure step not understood: {step.text}")
    name, inputs, outputs = (
        match.group(1),
        signature(match.group(2)),
        signature(match.group(3)),
    )
    table = [row for row in (step.table or []) if row != [""]]
    rows = table[1:] if table else []
    columns = [n for n, _ in inputs + outputs]
    if table and table[0] != columns:
        raise ValueError(f"procedure {name}: table {table[0]} is not {columns}")
    params = ", ".join(f"{n} {t}" for n, t in inputs)
    returns = ", ".join(f"{n} {t}" for n, t in outputs)
    records = ", ".join(
        "{"
        + ", ".join(
            f"p{i}: {gql_literal(tck_values.parse(c))}" for i, c in enumerate(row)
        )
        + "}"
        for row in rows
    )
    head = f"CREATE PROCEDURE `{name}`({params}) RETURNS ({returns}) AS {{ UNWIND [{records}] AS r"
    if not outputs:
        return head + " FINISH }"
    # A row answers the arguments it holds, a null argument the row's null:
    # the TCK's table matches by equivalence, where `=` gives null.
    filters = " AND ".join(
        f"(r.p{i} = ${n} OR (r.p{i} IS NULL AND ${n} IS NULL))"
        for i, (n, _) in enumerate(inputs)
    )
    picks = ", ".join(
        f"r.p{len(inputs) + i} AS {n}" for i, (n, _) in enumerate(outputs)
    )
    return head + (f" FILTER {filters}" if filters else "") + f" RETURN {picks} }}"


def values_table(table: list[list[str]]) -> list[str]:
    """The TCK table as `values:` block lines, a cell's pipes escaped."""
    for row in table[1:]:
        for cell in row:
            tck_values.parse(cell)
    return [
        "| " + " | ".join(c.replace("|", "\\|") for c in row) + " |" for row in table
    ]


def side_effects(table: list[list[str]] | None) -> list[str]:
    if not table:
        return ["side_effects: {}"]
    return ["side_effects:"] + [f"  {key}: {int(count)}" for key, count in table]


def neutral(text: str) -> str:
    """The text with each listed string literal replaced by its neutral value."""

    def swap(match: re.Match) -> str:
        digest = hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
        return NEUTRAL_VALUES.get(digest, match.group(0))

    return re.sub(r"'[^'\\]*'", swap, text)


def neutral_step(step: gherkin.Step) -> gherkin.Step:
    return gherkin.Step(
        step.text,
        None if step.docstring is None else neutral(step.docstring),
        None
        if step.table is None
        else [[neutral(c) for c in row] for row in step.table],
    )


def convert_scenario(scenario: gherkin.Scenario, graphs: dict[str, str]) -> list[Case]:
    """The cases of one scenario; raises ValueError on a step it does not know."""
    cases = [Case(slug(scenario.name), scenario.name)]
    case = cases[0]
    for original in scenario.steps:
        step = neutral_step(original)
        if step != original and NEUTRAL_NOTE not in case.comments:
            case.comments.append(NEUTRAL_NOTE)
        text = step.text
        if text in ("an empty graph", "any graph"):
            continue
        if (graph := re.fullmatch(r"the ([\w-]+) graph", text)) is not None:
            case.setup.append(graphs[graph.group(1)])
        elif text.startswith("having executed"):
            case.setup.append(step.docstring)
        elif text.startswith("parameters are"):
            case.params += [(n, param_text(tck_values.parse(v))) for n, v in step.table]
        elif text.startswith("there exists a procedure"):
            case.setup_gql.append(create_procedure(step))
        elif text.startswith("executing query"):
            case.query = step.docstring
        elif text.startswith("executing control query"):
            if case.params:
                raise ValueError("a control query after a query with parameters")
            follow = Case(f"{case.name}_control", f"{scenario.name} (control query)")
            follow.setup = case.setup + [case.query]
            follow.setup_gql = list(case.setup_gql)
            follow.query = step.docstring
            cases.append(follow)
            case = follow
        elif text == "the result should be empty":
            case.expect.append("empty: true")
        elif text.startswith("the result should be"):
            if "in order" in text:
                case.expect.append("ordered: true")
            if "ignoring element order for lists" in text:
                case.expect.append("unordered_lists: true")
            case.expect += ["values: |"] + [
                f"  {line}" for line in values_table(step.table)
            ]
        elif text == "no side effects":
            case.expect += side_effects(None)
        elif text.startswith("the side effects should be"):
            case.expect += side_effects(step.table)
        elif (error := ERROR.fullmatch(text)) is not None:
            kind, phase, detail = error.groups()
            static = "42" if phase == "compile time" else None
            status = DETAIL_CLASSES.get(detail) or static or ERROR_CLASSES[kind]
            case.comments.append(f"TCK: {kind} at {phase}: {detail}")
            if kind in PROVISIONAL:
                case.comments.append(f"the class of {kind} is provisional")
            case.expect += [f'gql_status: "{status}"'] + side_effects(None)
        else:
            raise ValueError(f"step not understood: {text}")
    return cases


def yaml_lines(case: Case) -> list[str]:
    lines = ["", f"  # {case.title}"] + [f"  # {c}" for c in case.comments]
    lines.append(f"  - name: {case.name}")
    if case.xfail:
        spec, issue, passes_in = case.xfail
        lines += [
            "    xfail:",
            f"      spec: {json.dumps(spec, ensure_ascii=False)}",
            f"      issue: {issue}",
        ]
        if passes_in:
            lines.append(f"      passes_in: [{', '.join(passes_in)}]")
    if case.runner_gap:
        gap = case.runner_gap
        lines += [
            "    runner_gap:",
            f"      runners: [{', '.join(gap['runners'])}]",
            f"      issue: {gap['issue']}",
            f"      reason: {json.dumps(gap['reason'])}",
        ]
    if case.setup_gql:
        lines += ["    setup_gql:"] + [
            f"      - {json.dumps(s, ensure_ascii=False)}" for s in case.setup_gql
        ]
    if case.setup:
        lines.append("    setup:")
        for statement in case.setup:
            lines.append("      - |")
            lines += [f"        {line}".rstrip() for line in statement.splitlines()]
    if case.params:
        lines += ["    params:"] + [f"      {n}: {v}" for n, v in case.params]
    lines.append("    query: |")
    lines += [f"      {line}".rstrip() for line in case.query.splitlines()]
    lines.append("    expect:")
    lines += [f"      {line}" for line in case.expect]
    return lines


def apply_gql_class(case: Case, entry: dict) -> None:
    """Expect GQL's class where a General Rule of GQL gives another than the TCK's."""
    old = next((e for e in case.expect if e.startswith("gql_status: ")), None)
    if old is None or not entry.get("note"):
        raise ValueError(f"gql_class on {case.name} needs an error case and a note")
    tck = old.removeprefix("gql_status: ").strip('"')
    case.expect[case.expect.index(old)] = f'gql_status: "{entry["gql_class"]}"'
    case.comments.append(
        f"GQL class {entry['gql_class']}, the TCK expects {tck}: {entry['note']}"
    )


def licence_header(text: str) -> list[str]:
    """The feature file's leading comment block, its licence and attribution."""
    out = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        out.append(line.rstrip())
    return out


def convert_feature(
    path: Path, features: Path, graphs: dict[str, str], known: dict, commit: str
) -> str | None:
    """The .gtest text for one feature file, or None when it has no case to run."""
    text = path.read_text(encoding="utf-8")
    feature = gherkin.parse_feature(text)
    cases: list[Case] = []
    left_out: list[tuple[str, str]] = []
    names: set[str] = set()
    for scenario in feature.scenarios:
        if "@ignore" in scenario.tags:
            left_out.append((scenario.name, "tagged @ignore upstream"))
            continue
        for case in convert_scenario(scenario, graphs):
            while case.name in names:
                case.name += "_"
            names.add(case.name)
            temporal_feature = "temporal" in path.relative_to(features).parts
            searched = case.query + "\n".join(case.setup) + "\n".join(case.expect)
            if temporal_feature or TEMPORAL.search(searched):
                case.runner_gap = TEMPORAL_GAP
            entry = known.get(case.name)
            if entry is not None and "gql_class" in entry:
                apply_gql_class(case, entry)
                rest = {
                    k: v for k, v in entry.items() if k not in ("gql_class", "note")
                }
                entry = rest or None
            if entry is None:
                cases.append(case)
            elif "issue" in entry:
                case.xfail = (
                    f"openCypher TCK {TAG} {path.stem}: {case.title}",
                    entry["issue"],
                    list(entry.get("passes_in", [])),
                )
                cases.append(case)
            elif entry.get("reason") in (
                "out-of-scope",
                "spec-disagreement",
            ) and entry.get("note"):
                left_out.append((case.title, f"{entry['reason']}: {entry['note']}"))
            else:
                raise ValueError(
                    f"known failure {path.stem}/{case.name} has no issue or reason"
                )
    stale = sorted(set(known) - names)
    if stale:
        raise ValueError(
            f"{path.stem}: known failures that name no case: {', '.join(stale)}"
        )
    if not cases:
        return None
    relative = path.relative_to(features).as_posix()
    lines = licence_header(text) + [
        "#",
        f"# Converted from the openCypher TCK, tck/features/{relative}, by",
        "# scripts/convert_opencypher_tck.py.",
        f"# Source: https://github.com/opencypher/openCypher at tag {TAG} ({commit})",
    ]
    if left_out:
        lines.append("# Not converted:")
        lines += [f"#   {name}: {reason}" for name, reason in left_out]
    lines += [
        "",
        "meta:",
        "  language: cypher",
        "  model: lpg",
        f'  section: "opencypher-tck.{path.stem}"',
        f"  title: {json.dumps(feature.name, ensure_ascii=False)}",
        "  dataset: empty",
        "  requires: [cypher]",
        "  values_dialect: tck",
        "",
        "tests:",
    ]
    for case in cases:
        lines += yaml_lines(case)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the openCypher TCK to .gtest files"
    )
    parser.add_argument(
        "--tck",
        type=Path,
        required=True,
        help="The tck folder of an openCypher checkout",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    features = args.tck / "features"
    graphs = {
        folder.name: (folder / f"{folder.name}.cypher")
        .read_text(encoding="utf-8")
        .strip()
        for folder in (args.tck / "graphs").iterdir()
        if folder.is_dir()
    }
    commit = (
        subprocess.run(
            ["git", "-C", str(args.tck), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "an unknown commit"
    )
    known_path = args.output / "known_failures.toml"
    known = (
        tomllib.loads(known_path.read_text(encoding="utf-8"))
        if known_path.exists()
        else {}
    )
    stems = set()
    written = 0
    for path in sorted(features.rglob("*.feature")):
        stems.add(path.stem)
        text = convert_feature(path, features, graphs, known.get(path.stem, {}), commit)
        if text is None:
            continue
        target = args.output / path.relative_to(features).with_suffix(".gtest")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        written += 1
    stale = sorted(set(known) - stems)
    if stale:
        raise SystemExit(
            f"known failures for features that do not exist: {', '.join(stale)}"
        )
    print(f"{written} of {len(stems)} features have cases -> {args.output}")


if __name__ == "__main__":
    main()
