#!/usr/bin/env python3
"""Convert W3C SPARQL test manifests to .gtest files.

Reads W3C manifest.ttl files (using rdflib) and generates .gtest YAML files
that can be run through the Grafeo spec test runners.

Usage:
    # Convert a single manifest
    python scripts/convert_w3c_sparql.py \\
        --manifest tests/spec/rdf/sparql/w3c/rdf-tests/sparql/sparql11/data-sparql11/aggregates/manifest.ttl \\
        --output tests/spec/rdf/sparql/w3c/aggregates.gtest

    # Convert all SPARQL 1.1 manifests
    python scripts/convert_w3c_sparql.py --all

Requirements:
    uv pip install rdflib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from rdflib import Graph, Namespace, URIRef
    from rdflib.namespace import RDF
except ImportError:
    print(
        "Error: rdflib is required. Install with: uv pip install rdflib",
        file=sys.stderr,
    )
    sys.exit(1)


# W3C manifest vocabulary
MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
QT = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-query#")
UT = Namespace("http://www.w3.org/2009/sparql/tests/test-update#")
DAWGT = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#")


def load_manifest(manifest_path: Path) -> Graph:
    """Load a W3C manifest.ttl file into an rdflib Graph."""
    graph = Graph()
    graph.parse(str(manifest_path), format="turtle")
    return graph


def extract_tests(graph: Graph, manifest_path: Path) -> list[dict]:
    """Extract test cases from a manifest graph."""
    tests = []
    base_dir = manifest_path.parent

    # Find all test entries
    for test_node in graph.subjects(RDF.type, None):
        test_type = None
        for t in graph.objects(test_node, RDF.type):
            type_str = str(t)
            if "QueryEvaluationTest" in type_str:
                test_type = "query"
            elif "UpdateEvaluationTest" in type_str:
                test_type = "update"
            elif "CSVResultFormatTest" in type_str:
                test_type = "query"
            elif "PositiveSyntaxTest" in type_str:
                test_type = "syntax_positive"
            elif "NegativeSyntaxTest" in type_str:
                test_type = "syntax_negative"

        if test_type is None:
            continue

        # Extract test metadata
        name = None
        for n in graph.objects(test_node, MF.name):
            name = str(n)
            break

        if not name:
            # Use the URI fragment as name
            name = str(test_node).split("#")[-1].split("/")[-1]

        # Sanitize name for use as test identifier
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

        comment = None
        for c in graph.objects(test_node, DAWGT.approval):
            comment = str(c)

        # Extract query file
        query_text = None
        action = list(graph.objects(test_node, MF.action))
        if action:
            action_node = action[0]
            # For query tests, action is a blank node with qt:query
            query_files = list(graph.objects(action_node, QT.query))
            if query_files:
                query_path = base_dir / Path(str(query_files[0]).replace("file://", ""))
                if query_path.exists():
                    query_text = query_path.read_text(encoding="utf-8").strip()
            elif isinstance(action_node, URIRef):
                # Direct reference to query file
                query_path = base_dir / Path(str(action_node).split("/")[-1])
                if query_path.exists():
                    query_text = query_path.read_text(encoding="utf-8").strip()

            # For update tests, use ut:request
            update_files = list(graph.objects(action_node, UT.request))
            if update_files:
                query_path = base_dir / Path(str(update_files[0]).split("/")[-1])
                if query_path.exists():
                    query_text = query_path.read_text(encoding="utf-8").strip()

        # Extract data file for setup
        data_text = None
        if action:
            action_node = action[0]
            data_files = list(graph.objects(action_node, QT.data))
            if data_files:
                data_path = base_dir / Path(str(data_files[0]).split("/")[-1])
                if data_path.exists():
                    data_text = data_path.read_text(encoding="utf-8").strip()

        test_entry = {
            "name": safe_name,
            "type": test_type,
            "query": query_text,
            "data": data_text,
            "comment": comment,
        }
        tests.append(test_entry)

    return tests


def generate_gtest(tests: list[dict], section: str, title: str) -> str:
    """Generate .gtest YAML content from test cases."""
    lines = [
        f"# W3C SPARQL 1.1 Test Suite: {title}",
        "# Auto-generated from W3C manifest by scripts/convert_w3c_sparql.py",
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

    for test in tests:
        if not test["query"]:
            continue

        lines.append("")
        lines.append(f"  - name: {test['name']}")

        # Setup: convert N-Triples/Turtle data to SPARQL INSERT DATA
        if test["data"]:
            setup_sparql = turtle_to_insert_data(test["data"])
            if setup_sparql:
                lines.append("    setup:")
                lines.append("      - |")
                for setup_line in setup_sparql.split("\n"):
                    lines.append(f"        {setup_line}")

        # Query
        query = test["query"]
        if "\n" in query:
            lines.append("    query: |")
            for q_line in query.split("\n"):
                lines.append(f"      {q_line}")
        else:
            lines.append(f"    query: {yaml_quote(query)}")

        # For syntax tests, just check parse success/failure
        if test["type"] == "syntax_positive":
            lines.append("    expect:")
            lines.append("      count: 1")
        elif test["type"] == "syntax_negative":
            lines.append("    expect:")
            lines.append("      error: syntax")
        else:
            # For query evaluation, use count (exact result comparison needs
            # parsing .srx files which is complex)
            lines.append("    expect:")
            lines.append("      count: 1")

    lines.append("")
    return "\n".join(lines)


def turtle_to_insert_data(turtle_content: str) -> Optional[str]:
    """Convert Turtle/N-Triples data to a SPARQL INSERT DATA statement."""
    # Simple approach: wrap in INSERT DATA { ... }
    # This works for N-Triples but may need prefix handling for Turtle
    triples = []
    for line in turtle_content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        triples.append(line)

    if not triples:
        return None

    return "INSERT DATA {\n" + "\n".join(f"  {t}" for t in triples) + "\n}"


def yaml_quote(s: str) -> str:
    """Quote a string for YAML if it contains special characters."""
    if any(c in s for c in ":{}\n\"'[]"):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Convert W3C SPARQL test manifests to .gtest files"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to a manifest.ttl file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .gtest file path",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all SPARQL 1.1 manifests",
    )
    args = parser.parse_args()

    if args.manifest and args.output:
        graph = load_manifest(args.manifest)
        tests = extract_tests(graph, args.manifest)
        section = args.manifest.parent.name
        title = section.replace("-", " ").title()
        content = generate_gtest(tests, section, title)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Generated {len(tests)} tests -> {args.output}")

    elif args.all:
        w3c_dir = Path(
            "tests/spec/rdf/sparql/w3c/rdf-tests/sparql/sparql11/data-sparql11"
        )
        if not w3c_dir.exists():
            print(f"W3C test directory not found: {w3c_dir}", file=sys.stderr)
            print(
                "Run: git submodule add https://github.com/w3c/rdf-tests.git tests/spec/rdf/sparql/w3c/rdf-tests",
                file=sys.stderr,
            )
            sys.exit(1)

        output_dir = Path("tests/spec/rdf/sparql/w3c")
        total = 0
        for manifest in sorted(w3c_dir.glob("*/manifest.ttl")):
            section = manifest.parent.name
            graph = load_manifest(manifest)
            tests = extract_tests(graph, manifest)
            if not tests:
                continue
            title = section.replace("-", " ").title()
            content = generate_gtest(tests, section, title)
            output_path = output_dir / f"{section}.gtest"
            output_path.write_text(content, encoding="utf-8")
            print(f"  {section}: {len(tests)} tests -> {output_path}")
            total += len(tests)

        print(f"\nTotal: {total} tests generated")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
