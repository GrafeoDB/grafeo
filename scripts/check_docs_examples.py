#!/usr/bin/env python3
"""Check that the documentation's examples work.

Every Python block on a page runs, in order, in one fresh subprocess. Every
query block (GQL, Cypher, SPARQL, Gremlin, SQL/PGQ) is split into statements
and each statement is parsed through the binding's EXPLAIN entry points.

    python scripts/check_docs_examples.py <docs dir> [--section <subdir or page>]
                                          [--report <file.json>]

Markers (HTML comments, invisible when rendered):

    <!-- docs-check: skip <reason> -->     on the line before a fence
    <!-- docs-check: setup                  hidden Python, runs before the
    import grafeo                           page's first Python block
    -->

Exit code 0 when no block fails. A skip marker without a reason, and a page
whose checkable blocks are all skipped, fail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PYTHON_LANGS = {"python", "py"}
QUERY_LANGS = {"gql", "cypher", "sparql", "gremlin", "graphql", "sql"}
PAGE_TIMEOUT = 180

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$")
_SKIP = re.compile(r"^[ \t]*<!--\s*docs-check:\s*skip\b(?P<reason>.*?)-->[ \t]*$")
_SETUP_OPEN = re.compile(r"^[ \t]*<!--\s*docs-check:\s*setup\b(?P<rest>.*)$")


class MarkerError(ValueError):
    """A docs-check marker that is malformed (for example a skip without a reason)."""


@dataclass
class Block:
    lang: str
    code: str
    line: int  # 1-based page line of the opening fence (of the marker for setup)
    skip: str | None = None
    setup: bool = False


def extract_blocks(text: str) -> list[Block]:
    """Find fenced blocks and markers at any indentation. CRLF reads like LF."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        setup = _SETUP_OPEN.match(raw)
        if setup:
            start = i
            if setup.group("rest").strip():
                raise MarkerError(
                    f"line {i + 1}: setup code goes on the lines after the marker"
                )
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "-->":
                body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise MarkerError(
                    f"line {start + 1}: setup marker is never closed with -->"
                )
            blocks.append(Block("python", _dedent(body), start + 1, None, True))
            i += 1
            continue
        skip = _SKIP.match(raw)
        if skip:
            reason = skip.group("reason").strip()
            if not reason:
                raise MarkerError(f"line {i + 1}: skip marker needs a reason")
            if i + 1 >= len(lines) or not _FENCE.match(lines[i + 1]):
                raise MarkerError(
                    f"line {i + 1}: skip marker must sit on the line before a fence"
                )
            i += 1
            blocks.append(_read_fence(lines, i, reason))
            i = _fence_end(lines, i) + 1
            continue
        if _FENCE.match(raw):
            blocks.append(_read_fence(lines, i, None))
            i = _fence_end(lines, i) + 1
            continue
        i += 1
    return blocks


def _fence_end(lines: list[str], start: int) -> int:
    m = _FENCE.match(lines[start])
    assert m
    ch, n = m.group("fence")[0], len(m.group("fence"))
    for j in range(start + 1, len(lines)):
        c = _FENCE.match(lines[j])
        if (
            c
            and c.group("fence")[0] == ch
            and len(c.group("fence")) >= n
            and not c.group("info").strip()
        ):
            return j
    return len(lines) - 1


def _read_fence(lines: list[str], start: int, skip: str | None) -> Block:
    m = _FENCE.match(lines[start])
    assert m
    info = m.group("info").strip()
    lang = re.split(r"[\s{]", info, maxsplit=1)[0].lower().lstrip(".") if info else ""
    end = _fence_end(lines, start)
    return Block(
        lang,
        _dedent(lines[start + 1 : end], len(m.group("indent"))),
        start + 1,
        skip,
        False,
    )


def _dedent(body: list[str], indent: int | None = None) -> str:
    if indent is None:
        indent = min((len(b) - len(b.lstrip()) for b in body if b.strip()), default=0)
    out = []
    for b in body:
        k = 0
        while k < indent and k < len(b) and b[k] in " \t":
            k += 1
        out.append(b[k:])
    return "\n".join(out).strip("\n")


_COMMENTS = {
    "gql": ("--", "//"),
    "sql": ("--",),
    "cypher": ("//",),
    "gremlin": ("//",),
    "sparql": ("#",),
}


def split_statements(code: str, lang: str) -> list[str]:
    """Split a query block into statements.

    Statements end at a blank line, or at a `;` outside a quoted string (not
    for SPARQL, where `;` is predicate-list syntax). A comment is dropped when
    it starts a line or follows a `;`; `--` inside a pattern and `#` inside an
    IRI stay put. Fragments that are empty or only comments are dropped.
    """
    code = code.replace("\r\n", "\n")
    comments = _COMMENTS.get(lang, ())
    split_semicolon = lang != "sparql"
    chunks: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    line_start = True  # only whitespace so far on this line
    after_sep = False  # only whitespace since the last `;` on this line

    def flush() -> None:
        text = "".join(cur).strip()
        if text:
            chunks.append(text)
        cur.clear()

    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if quote:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(code[i + 1])
                i += 1
            elif c == quote:
                if code[i + 1 : i + 2] == quote:
                    cur.append(quote)
                    i += 1
                else:
                    quote = None
            i += 1
            continue
        if (line_start or after_sep) and any(code.startswith(t, i) for t in comments):
            end = code.find("\n", i)
            i = n if end < 0 else end
            line_start = False
            continue
        if c == "\n":
            if line_start:
                flush()
            else:
                cur.append(c)
            line_start, after_sep = True, False
        elif c == ";" and split_semicolon:
            flush()
            after_sep, line_start = True, False
        elif c in " \t\r":
            cur.append(c)
        else:
            if c in "'\"`":
                quote = c
            cur.append(c)
            line_start = after_sep = False
        i += 1
    flush()
    if lang != "sparql":
        return chunks

    header: list[str] = []
    result: list[str] = []
    for chunk in chunks:
        rest = chunk.split("\n")
        while rest and re.match(r"^\s*(PREFIX|BASE)\b", rest[0], re.I):
            header.append(rest.pop(0).strip())
        if rest:
            result.append("\n".join([*header, *rest]))
    return result


def query_kind(page: str, lang: str) -> str | None:
    """Which checker a fenced language maps to on a page; None means unchecked."""
    page = page.replace("\\", "/")
    if lang == "gql":
        return "gql"
    if lang == "sql":
        return "sql" if "model-and-query/sql-pgq/" in page else "gql"
    if lang in ("cypher", "sparql", "gremlin"):
        return lang
    # graphql: the binding has no parse-only entry point, so it stays unchecked.
    return None


_EXPLAIN = {
    "gql": "explain",
    "cypher": "explain_cypher",
    "sparql": "explain_sparql",
    "sql": "explain_sql",
    "gremlin": "explain_gremlin",
}
_EXECUTE = {
    "gql": "execute",
    "cypher": "execute_cypher",
    "sparql": "execute_sparql",
    "sql": "execute_sql",
    "gremlin": "execute_gremlin",
}


def _code_of(err: Exception) -> str:
    m = re.search(r"GRAFEO-([A-Z]\d{3})", str(err))
    return m.group(1) if m else ""


# Q002 passes only when the message names state the page does not set up: a
# graph, label, property, index or type that does not exist / is not found, or
# a feature that needs an index. Every other Q002 (undefined variable, unknown
# function, wrong arity, non-grouping key, ...) is a real mistake in the example.
_MISSING_STATE = re.compile(
    r"\b(?:graph|label|property|index|type|schema|procedure)\b[^\n]*"
    r"\b(?:does not exist|not found|is not defined|unknown)\b"
    r"|\bunknown\s+(?:graph|label|property|index|type|schema|procedure)\b"
    r"|\b(?:requires|needs)\s+an?\s+(?:\w+\s+)?index\b|\bno\s+(?:\w+\s+)?index\b",
    re.IGNORECASE,
)
_LEADING_EXPLAIN = re.compile(
    r"^\s*(?:EXPLAIN(?:\s+ANALYZE)?|PROFILE)\s+", re.IGNORECASE
)


def check_statement(db, kind: str, stmt: str) -> str | None:
    """Error text, or None when the statement parses.

    A leading EXPLAIN [ANALYZE] or PROFILE is dropped first. Q001 (syntax)
    fails. Q002 passes only for missing state (see `_MISSING_STATE`). A
    statement EXPLAIN refuses (X001 such as schema DDL, or a transaction or
    session command that reports Q002) runs for real on a fresh in-memory
    database and fails only on a syntax error.
    """
    while _LEADING_EXPLAIN.match(stmt):
        stmt = _LEADING_EXPLAIN.sub("", stmt, count=1)
    try:
        getattr(db, _EXPLAIN[kind])(stmt)
        return None
    except Exception as err:  # noqa: BLE001 - the binding raises its own types
        code, text = _code_of(err), str(err)
        if code == "Q001":
            return text
        refused = code == "X001" or (
            code == "Q002" and "transaction or session" in text
        )
        if not refused:
            if code == "Q002" and _MISSING_STATE.search(text):
                return None
            return text
    import grafeo  # noqa: PLC0415

    fresh = grafeo.GrafeoDB()
    try:
        getattr(fresh, _EXECUTE[kind])(stmt)
    except Exception as err:  # noqa: BLE001
        if _code_of(err) == "Q001":
            return str(err)
    return None


_DRIVER = r"""
import json, sys, traceback
spec = json.load(open(sys.argv[1], encoding="utf-8"))
ns = {"__name__": "__docs__"}
errors = []
for b in spec:
    fname = "<docs-block>"
    try:
        exec(compile(b["code"], fname, "exec"), ns)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        lineno = getattr(e, "lineno", None) if isinstance(e, SyntaxError) else None
        if lineno is None:
            for fr in traceback.extract_tb(e.__traceback__):
                if fr.filename == fname:
                    lineno = fr.lineno
        errors.append({"line": b["line"] + (lineno or 0),
                       "error": "".join(traceback.format_exception_only(type(e), e)).strip()})
json.dump(errors, open(sys.argv[2], "w", encoding="utf-8"))
"""


def run_python_page(blocks: list[Block]) -> list[str]:
    """Run setup, then the page's Python blocks in one fresh subprocess."""
    setups = [b for b in blocks if b.setup]
    body = [
        b for b in blocks if not b.setup and b.lang in PYTHON_LANGS and b.skip is None
    ]
    if not body:
        return []
    spec = [{"code": b.code, "line": b.line} for b in [*setups, *body]]
    with tempfile.TemporaryDirectory() as tmp:
        spec_file, out_file = Path(tmp, "spec.json"), Path(tmp, "out.json")
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "MPLBACKEND": "Agg",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _DRIVER, str(spec_file), str(out_file)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                timeout=PAGE_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [
                f"line {body[0].line}: the page's Python blocks ran over {PAGE_TIMEOUT}s"
            ]
        if not out_file.exists():
            tail = " ".join(
                (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
            )
            return [
                f"line {body[0].line}: Python subprocess died ({proc.returncode}) {tail}"
            ]
        found = json.loads(out_file.read_text(encoding="utf-8"))
    return [f"line {e['line']}: {e['error']}" for e in found]


def check_page(rel: str, text: str) -> dict:
    result: dict = {
        "page": rel,
        "ran": 0,
        "checked": 0,
        "skipped": [],
        "unchecked": {},
        "failures": [],
    }
    try:
        blocks = extract_blocks(text)
    except MarkerError as err:
        result["failures"].append({"line": 0, "lang": "marker", "error": str(err)})
        return result
    real = [b for b in blocks if not b.setup]
    result["ran"] = sum(1 for b in real if b.lang in PYTHON_LANGS and b.skip is None)
    for e in run_python_page(blocks):
        m = re.match(r"line (\d+): (.*)", e, re.S)
        result["failures"].append(
            {
                "line": int(m.group(1)) if m else 0,
                "lang": "python",
                "error": m.group(2) if m else e,
            }
        )
    db = None
    for b in real:
        if b.skip is not None:
            result["skipped"].append({"line": b.line, "lang": b.lang, "reason": b.skip})
            continue
        if b.lang in PYTHON_LANGS:
            continue
        kind = query_kind(rel, b.lang) if b.lang in QUERY_LANGS else None
        if kind is None:
            name = b.lang or "(none)"
            result["unchecked"][name] = result["unchecked"].get(name, 0) + 1
            continue
        if db is None:
            import grafeo  # noqa: PLC0415

            db = grafeo.GrafeoDB()
            missing = [m for m in _EXPLAIN.values() if not hasattr(db, m)]
            if missing:
                raise SystemExit(
                    f"error: the installed grafeo {getattr(grafeo, '__version__', '?')} lacks {', '.join(missing)}; install a newer wheel"
                )
        result["checked"] += 1
        for stmt in split_statements(b.code, b.lang):
            err = check_statement(db, kind, stmt)
            if err:
                result["failures"].append(
                    {
                        "line": b.line,
                        "lang": b.lang,
                        "error": f"{stmt.splitlines()[0][:80]}: {err}",
                    }
                )
                break
    skipped_checkable = sum(
        1
        for s in result["skipped"]
        if s["lang"] in PYTHON_LANGS or query_kind(rel, s["lang"]) is not None
    )
    if skipped_checkable and not (result["ran"] or result["checked"]):
        result["failures"].append(
            {
                "line": 0,
                "lang": "marker",
                "error": f"every checkable block on the page is skipped ({skipped_checkable})",
            }
        )
    return result


def _pages(docs: Path, section: str | None) -> list[Path]:
    root = docs / section if section else docs
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"error: {root} is neither a directory nor a page")
    return sorted(root.rglob("*.md"))


def _section_of(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else "(root)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check that the docs' Python and query examples work."
    )
    ap.add_argument("docs", type=Path)
    ap.add_argument(
        "--section", help="a subdirectory or a single page, relative to the docs dir"
    )
    ap.add_argument("--report", type=Path, help="write a JSON report")
    args = ap.parse_args(argv)

    pages = []
    for path in _pages(args.docs, args.section):
        rel = path.relative_to(args.docs).as_posix()
        res = check_page(rel, path.read_bytes().decode("utf-8"))
        pages.append(res)
        if not (
            res["ran"]
            or res["checked"]
            or res["skipped"]
            or res["unchecked"]
            or res["failures"]
        ):
            continue
        skipped = str(len(res["skipped"]))
        if res["skipped"]:
            skipped += f" ({'; '.join(sorted({s['reason'] for s in res['skipped']}))})"
        unchecked = str(sum(res["unchecked"].values()))
        if res["unchecked"]:
            unchecked += f" ({', '.join(f'{k} x{v}' for k, v in sorted(res['unchecked'].items()))})"
        print(
            f"{rel}: ran {res['ran']} / checked {res['checked']} / skipped {skipped} / unchecked {unchecked}"
        )
        for f in res["failures"]:
            first = f["error"].splitlines()[0] if f["error"] else ""
            print(f"  FAIL line {f['line']} [{f['lang']}]: {first}")

    sections: dict[str, dict] = {}
    for p in pages:
        s = sections.setdefault(
            _section_of(p["page"]),
            {"ran": 0, "checked": 0, "skipped": 0, "unchecked": 0, "failing": 0},
        )
        s["ran"] += p["ran"]
        s["checked"] += p["checked"]
        s["skipped"] += len(p["skipped"])
        s["unchecked"] += sum(p["unchecked"].values())
        s["failing"] += len(p["failures"])
    failing = sum(s["failing"] for s in sections.values())
    print(f"\n{failing} failing block(s)")
    for name, s in sorted(sections.items()):
        print(
            f"  {name}: ran {s['ran']} / checked {s['checked']} / skipped {s['skipped']}"
            f" / unchecked {s['unchecked']} / failing {s['failing']}"
        )
    if args.report:
        payload = (
            json.dumps(
                {"failing": failing, "sections": sections, "pages": pages}, indent=2
            )
            + "\n"
        )
        args.report.write_bytes(payload.encode("utf-8"))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
