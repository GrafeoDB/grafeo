#!/usr/bin/env python3
"""Convert W3C SPARQL test manifests to .gtest files.

Reads a W3C manifest.ttl (with rdflib) and writes one .gtest file of its query
evaluation tests, in the manifest's own order. Each case carries the result the
manifest names:

- the data files become one `INSERT DATA` of N-Triples, so prefixes and Turtle
  shorthand need no handling by the runners;
- a SELECT result (.srx, .srj, .tsv) becomes `terms` rows, or `empty`;
  an ASK result becomes a `true` or `false` row. Rows compare as a sorted bag,
  so `ORDER BY` is not checked.

A test the .gtest format cannot express (named graphs, updates, syntax-only
tests, CSV format tests, entailment, graph results) is not written with a
guessed expectation: it is left out and listed with its reason, in the file's
header and on stderr.

`tests/spec/rdf/sparql/w3c/known_failures.toml` lists, by section, the cases
Grafeo does not pass: each becomes a strict `xfail:` naming its issue, or, for a
case no fix of Grafeo can pass as written, is left out as a `spec-disagreement`.
Each file's header names the rdf-tests commit it was generated from and the
W3C licence.

Usage:
    # Convert a single manifest
    uv run --with rdflib python scripts/convert_w3c_sparql.py \\
        --manifest tests/spec/rdf/sparql/w3c/rdf-tests/sparql/sparql11/aggregates/manifest.ttl \\
        --output tests/spec/rdf/sparql/w3c/aggregates.gtest

    # Convert all SPARQL 1.1 manifests
    uv run --with rdflib python scripts/convert_w3c_sparql.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

try:
    import rdflib
    from rdflib import BNode, Graph, Literal, Namespace, URIRef
    from rdflib.collection import Collection
    from rdflib.compare import to_canonical_graph
    from rdflib.namespace import RDF, XSD
    from rdflib.query import Result
    from rdflib.util import guess_format
except ImportError:
    print(
        "Error: rdflib is required. Run with: uv run --with rdflib python scripts/convert_w3c_sparql.py",
        file=sys.stderr,
    )
    sys.exit(1)

# Keep every literal as the suite writes it: rdflib would rewrite "3.21E4"^^xsd:double as "32100.0"
rdflib.NORMALIZE_LITERALS = False


# W3C manifest vocabulary
MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
QT = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-query#")
DAWGT = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#")
SD = Namespace("http://www.w3.org/ns/sparql-service-description#")

# Where the suite publishes its files; a data file's relative IRIs resolve against it
PUBLISHED_BASE = "http://www.w3.org/2009/sparql/docs/tests/data-sparql11/"

# Result file suffix to rdflib result parser (CSV drops term types, so it is not one)
RESULT_FORMATS = {".srx": "xml", ".srj": "json", ".tsv": "tsv"}


@dataclass
class Case:
    """One converted test: everything a .gtest case needs."""

    name: str
    title: str
    query: str
    data: list[str]
    expect: list[str] = field(default_factory=list)
    xfail: tuple[str, str] | None = None


@dataclass
class Converted:
    """The converted cases of one manifest, the tests left out with their reason, and
    the suite's commit."""

    cases: list[Case]
    left_out: list[tuple[str, str]]
    commit: str | None = None


def suite_commit(manifest: Path) -> str | None:
    """The commit of the rdf-tests checkout that holds the manifest, if it is one."""
    result = subprocess.run(
        ["git", "-C", str(manifest.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def load_known_failures(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """The known failures by section (a manifest's folder name), then case name."""
    if not path.exists():
        return {}
    with path.open("rb") as source:
        return tomllib.load(source)


def file_path(uri: object) -> Path:
    """The local path of a `file:` IRI, on Windows as well (`file:///C:/x` is `C:\\x`)."""
    return Path(url2pathname(urlparse(str(uri)).path))


def indented(line: str) -> str:
    """A query line with its leading tabs as spaces; a tab inside it may be in a literal."""
    text = line.lstrip(" \t")
    return line[: len(line) - len(text)].expandtabs(4) + text


def comment_title(name: str) -> str:
    """A test's name on one comment line, its ` -- ` separators written as `: `."""
    return re.sub(r"\s+-{2,}\s+", ": ", " ".join(name.split()))


def slug(text: str) -> str:
    """A snake_case test name: lowercase letters, digits and underscores."""
    name = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return name if name and not name[0].isdigit() else f"t_{name}"


def ntriples(paths: list[Path]) -> list[str]:
    """The triples of the data files as sorted N-Triples lines, the same on every run.

    Relative IRIs resolve against the file's published location, not the local
    checkout. rdflib gives blank nodes fresh labels on every parse, so they get
    canonical labels first, renamed `b0`, `b1`, ... in the canonical order.
    """
    graph = Graph()
    for path in paths:
        public = f"{PUBLISHED_BASE}{path.parent.name}/{path.name}"
        graph.parse(path, format=guess_format(str(path)) or "turtle", publicID=public)
    labels: dict[BNode, BNode] = {}

    def label(node: object) -> object:
        if isinstance(node, BNode):
            return labels.setdefault(node, BNode(f"b{len(labels)}"))
        return node

    relabeled = Graph()
    canonical = to_canonical_graph(graph)
    for s, p, o in sorted(canonical, key=lambda triple: [t.n3() for t in triple]):
        relabeled.add((label(s), p, label(o)))
    text = relabeled.serialize(format="nt")
    return sorted(line for line in text.splitlines() if line.strip())


def term(value: object) -> str:
    """A result cell in the `terms` form, as a YAML single-quoted scalar."""
    if value is None:
        return "null"
    if isinstance(value, BNode):
        text = "_:b"
    elif isinstance(value, URIRef):
        text = f"<{value}>"
    elif isinstance(value, Literal):
        lexical = json.dumps(str(value), ensure_ascii=False)
        if value.language:
            text = f"{lexical}@{value.language}"
        elif value.datatype and value.datatype != XSD.string:
            text = f"{lexical}^^<{value.datatype}>"
        else:
            text = lexical
    else:
        raise ValueError(f"unexpected result term {value!r}")
    return "'" + text.replace("'", "''") + "'"


def column_order(query: str, variables: list[str]) -> list[str]:
    """The result's variables in the order of their first appearance in the query.

    A result file's header may list the variables in any order (W3C compares
    by name), while a .gtest compares columns by position. First appearance
    is the SELECT clause's order, and for `SELECT *` the pattern's order.
    """
    first: dict[str, int] = {}
    for match in re.finditer(r"[?$]([A-Za-z0-9_]+)", query):
        first.setdefault(match.group(1), match.start())
    return sorted(variables, key=lambda var: first.get(var, len(query)))


def expectation(path: Path, query: str) -> list[str]:
    """The `expect:` lines for a result file."""
    with path.open("rb") as source:
        result = Result.parse(source, format=RESULT_FORMATS[path.suffix])
    if result.type == "ASK":
        return ["rows:", f"  - [{str(result.askAnswer).lower()}]"]
    if not result.bindings:
        return ["empty: true"]
    variables = column_order(query, [str(var) for var in result.vars])
    bindings = [
        {str(var): value for var, value in row.items()} for row in result.bindings
    ]
    rows = (", ".join(term(row.get(var)) for var in variables) for row in bindings)
    return ["terms:"] + [f"  - [{row}]" for row in rows]


def why_left_out(graph: Graph, test: URIRef, action: object) -> str | None:
    """The reason a test cannot become a .gtest case, or None when it can."""
    kinds = {str(kind).split("#")[-1] for kind in graph.objects(test, RDF.type)}
    if graph.value(test, DAWGT.approval) == DAWGT.Withdrawn:
        return "withdrawn"
    if "QueryEvaluationTest" not in kinds:
        return f"{', '.join(sorted(kinds))}: not a query evaluation test"
    requires = graph.value(test, MF.requires)
    if requires is not None:
        return f"requires {str(requires).split('#')[-1]}"
    if graph.value(action, SD.entailmentRegime) is not None:
        return "entailment regime"
    if graph.value(action, QT.graphData) is not None:
        return "named graphs (qt:graphData)"
    if graph.value(action, QT.serviceData) is not None:
        return "federated service data"
    query = graph.value(action, QT.query)
    if query is None or not file_path(query).exists():
        return "query file missing"
    result = graph.value(test, MF.result)
    if result is None or not file_path(result).exists():
        return "result file missing"
    if file_path(result).suffix not in RESULT_FORMATS:
        return f"{file_path(result).suffix} result (a graph, not a result set)"
    return None


def convert_manifest(
    manifest: Path, known: dict[str, dict[str, str]] | None = None
) -> Converted:
    """Convert every test of a manifest, in the order of its `mf:entries`.

    `known` is the section's known failures: a case listed with an `issue` becomes
    an expected failure held against its W3C test; one listed as a
    `spec-disagreement` is left out. An entry that names no converted case is an
    error, so the list cannot go stale.
    """
    known = known or {}
    graph = Graph()
    graph.parse(manifest, format="turtle")
    entries = next(graph.objects(None, MF.entries), None)
    tests = list(Collection(graph, entries)) if entries is not None else []

    cases: list[Case] = []
    left_out: list[tuple[str, str]] = []
    names: set[str] = set()
    for test in tests:
        label = str(test).split("#")[-1]
        action = graph.value(test, MF.action)
        reason = why_left_out(graph, test, action)
        if reason:
            left_out.append((label, reason))
            continue
        name = slug(label)
        while name in names:
            name += "_"
        names.add(name)
        entry = known.get(name)
        xfail = None
        if entry is not None:
            if "issue" in entry:
                xfail = (
                    f"W3C SPARQL 1.1 {manifest.parent.name}/{label}",
                    entry["issue"],
                )
            elif entry.get("reason") == "spec-disagreement" and entry.get("note"):
                left_out.append((label, f"spec-disagreement: {entry['note']}"))
                continue
            else:
                raise ValueError(
                    f"known failure {name}: give an `issue`, or "
                    '`reason = "spec-disagreement"` and a `note`'
                )
        data = [file_path(d) for d in graph.objects(action, QT.data)]
        query = (
            file_path(graph.value(action, QT.query)).read_text(encoding="utf-8").strip()
        )
        cases.append(
            Case(
                name=name,
                title=comment_title(str(graph.value(test, MF.name) or label)),
                query=query,
                data=ntriples(data) if data else [],
                expect=expectation(file_path(graph.value(test, MF.result)), query),
                xfail=xfail,
            )
        )
    stale = sorted(set(known) - names)
    if stale:
        raise ValueError(
            f"known failures that name no converted case: {', '.join(stale)}"
        )
    return Converted(cases, left_out, suite_commit(manifest))


def generate_gtest(converted: Converted, section: str, title: str) -> str:
    """The .gtest file for one converted manifest."""
    commit = converted.commit or "an unknown commit"
    lines = [
        f"# W3C SPARQL 1.1 Test Suite: {title}",
        "# Generated from the W3C manifest by scripts/convert_w3c_sparql.py",
        f"# Source: https://github.com/w3c/rdf-tests at {commit}",
        "# Copyright W3C (MIT, ERCIM, Keio, Beihang), used under the W3C test suite",
        "# licences: https://www.w3.org/Consortium/Legal/2008/04-testsuite-copyright.html",
    ]
    if converted.left_out:
        lines.append("# Not converted:")
        lines += [f"#   {label}: {reason}" for label, reason in converted.left_out]
    lines += [
        "",
        "meta:",
        "  language: sparql",
        "  model: rdf",
        f'  section: "w3c.{section}"',
        f"  title: W3C {title}",
        "  dataset: empty",
        "  requires: [sparql, rdf]",
        "",
        "tests:",
    ]
    for case in converted.cases:
        lines += ["", f"  # {case.title}", f"  - name: {case.name}"]
        if case.xfail:
            spec, issue = case.xfail
            lines += [
                "    xfail:",
                f"      spec: {json.dumps(spec)}",
                f"      issue: {issue}",
            ]
        if case.data:
            lines += ["    setup:", "      - |", "        INSERT DATA {"]
            lines += [f"          {triple}" for triple in case.data]
            lines += ["        }"]
        lines.append("    query: |")
        lines += [
            f"      {indented(line)}".rstrip() for line in case.query.splitlines()
        ]
        lines.append("    expect:")
        lines += [f"      {line}" for line in case.expect]
    lines.append("")
    return "\n".join(lines)


def write_gtest(manifest: Path, output: Path, known_failures: Path) -> None:
    """Convert one manifest to one .gtest file and report what was left out."""
    section = manifest.parent.name
    converted = convert_manifest(
        manifest, load_known_failures(known_failures).get(section)
    )
    title = section.replace("-", " ").title()
    for label, reason in converted.left_out:
        print(f"    not converted: {label}: {reason}", file=sys.stderr)
    if not converted.cases:
        print(f"  {section}: no case to convert, nothing written")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        generate_gtest(converted, section, title), encoding="utf-8", newline="\n"
    )
    print(f"  {section}: {len(converted.cases)} cases -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert W3C SPARQL test manifests to .gtest files"
    )
    parser.add_argument("--manifest", type=Path, help="Path to a manifest.ttl file")
    parser.add_argument("--output", type=Path, help="Output .gtest file path")
    parser.add_argument(
        "--all", action="store_true", help="Convert all SPARQL 1.1 manifests"
    )
    parser.add_argument(
        "--known-failures",
        type=Path,
        default=Path("tests/spec/rdf/sparql/w3c/known_failures.toml"),
        help="The expected failures and left-out cases, by section",
    )
    args = parser.parse_args()

    if args.manifest and args.output:
        write_gtest(args.manifest, args.output, args.known_failures)
    elif args.all:
        w3c_dir = Path("tests/spec/rdf/sparql/w3c/rdf-tests/sparql/sparql11")
        if not w3c_dir.exists():
            print(f"W3C test directory not found: {w3c_dir}", file=sys.stderr)
            print(
                "Run: git clone https://github.com/w3c/rdf-tests.git tests/spec/rdf/sparql/w3c/rdf-tests",
                file=sys.stderr,
            )
            sys.exit(1)
        for manifest in sorted(w3c_dir.glob("*/manifest.ttl")):
            write_gtest(
                manifest,
                Path("tests/spec/rdf/sparql/w3c") / f"{manifest.parent.name}.gtest",
                args.known_failures,
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
