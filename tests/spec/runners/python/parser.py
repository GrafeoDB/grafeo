"""Parse .gtest YAML files into structured test cases.

Mirrors the Rust build.rs parser in crates/grafeo-spec-tests/build.rs so that
the Python runner exercises the exact same test definitions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Try PyYAML first, fall back to our own line-based parser
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Meta:
    language: str = "gql"
    model: str = ""
    section: str = ""
    title: str = ""
    dataset: str = "empty"
    requires: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    iso: list[str] = field(default_factory=list)


@dataclass
class Expect:
    rows: list[list[str]] = field(default_factory=list)
    ordered: bool = False
    count: int | None = None
    empty: bool = False
    error: str | None = None
    hash: str | None = None
    precision: int | None = None
    columns: list[str] = field(default_factory=list)


@dataclass
class TestCase:
    name: str = ""
    query: str | None = None
    statements: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)
    # Each value as the file writes it, quotes included: whether it was quoted
    # decides its type (see ``coerce_params``).
    params: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    iso: list[str] = field(default_factory=list)
    skip: str | None = None
    expect: Expect = field(default_factory=Expect)
    variants: dict[str, str] = field(default_factory=dict)
    language: str | None = None
    dataset: str | None = None


@dataclass
class GtestFile:
    meta: Meta
    tests: list[TestCase]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_gtest_file(path: Path) -> GtestFile:
    """Parse a .gtest file and return a GtestFile with meta + test cases.

    Tries PyYAML first but falls back to the line-based parser when the
    file contains constructs that are valid in .gtest but not in strict YAML
    (e.g. unquoted strings with embedded colons).
    """
    content = path.read_text(encoding="utf-8")
    if HAS_YAML:
        try:
            return _parse_with_yaml(content, path)
        except (_yaml.YAMLError, TypeError):
            # .gtest files may contain bare strings with colons that YAML
            # rejects: the line-based parser reads them.
            return _parse_line_based(content, path)
    return _parse_line_based(content, path)


# ---------------------------------------------------------------------------
# YAML-based parser (preferred)
# ---------------------------------------------------------------------------


def _parse_with_yaml(content: str, path: Path) -> GtestFile:
    data = _yaml.safe_load(content)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping at top level in {path}")

    meta = _parse_meta_dict(data.get("meta", {}))
    raw_tests = data.get("tests", [])
    param_texts = _yaml_param_texts(content)
    tests: list[TestCase] = []
    for position, raw in enumerate(raw_tests):
        tc = _parse_test_dict(raw)
        tc.params = param_texts[position]
        tests.append(tc)
    return GtestFile(meta=meta, tests=tests)


def _yaml_param_texts(content: str) -> list[dict[str, str]]:
    """The params of each test as the file writes them, quotes included.

    ``safe_load`` types a value by YAML's rules (``0x1F`` and ``1_000`` become
    ints, a quoted ``"[3]"`` loses its quotes, a surrogate pair stays two
    halves), so the value text is cut from the source at the node's position
    and typed by ``coerce_params``, as the other runners do.
    """
    root = _yaml.compose(content, Loader=_yaml.SafeLoader)
    tests_node = _mapping_value(root, "tests")
    if not isinstance(tests_node, _yaml.SequenceNode):
        return []
    texts: list[dict[str, str]] = []
    for test_node in tests_node.value:
        params: dict[str, str] = {}
        params_node = _mapping_value(test_node, "params")
        if isinstance(params_node, _yaml.MappingNode):
            for key_node, value_node in params_node.value:
                start = value_node.start_mark.index
                end = value_node.end_mark.index
                params[str(key_node.value)] = content[start:end].strip()
        texts.append(params)
    return texts


def _mapping_value(node, key: str):
    """The value node under ``key`` in a YAML mapping node, or None."""
    if not isinstance(node, _yaml.MappingNode):
        return None
    for key_node, value_node in node.value:
        if isinstance(key_node, _yaml.ScalarNode) and key_node.value == key:
            return value_node
    return None


def _parse_meta_dict(d: dict) -> Meta:
    if d is None:
        return Meta()
    m = Meta()
    m.language = str(d.get("language", "gql"))
    m.model = str(d.get("model", ""))
    m.section = str(d.get("section", ""))
    m.title = str(d.get("title", ""))
    m.dataset = str(d.get("dataset", "empty"))
    m.requires = _as_string_list(d.get("requires", []))
    m.tags = _as_string_list(d.get("tags", []))
    m.iso = _as_string_list(d.get("iso", []))
    return m


def _parse_test_dict(d: dict) -> TestCase:
    tc = TestCase()
    tc.name = str(d.get("name", ""))
    tc.skip = d.get("skip")
    if tc.skip is not None:
        tc.skip = str(tc.skip)
    tc.tags = _as_string_list(d.get("tags", []))
    tc.requires = _as_string_list(d.get("requires", []))
    tc.iso = _as_string_list(d.get("iso", []))

    # query: may be a string or a multi-line block
    q = d.get("query")
    if q is not None:
        tc.query = str(q).strip()

    # setup / statements: lists of strings
    tc.setup = _as_string_list(d.get("setup", []))
    tc.statements = _as_string_list(d.get("statements", []))

    # params: the caller sets them from the source text (_yaml_param_texts)

    # per-test language override (e.g. "graphql-rdf")
    lang = d.get("language")
    if lang is not None:
        tc.language = str(lang)

    # per-test dataset override
    ds = d.get("dataset")
    if ds is not None:
        tc.dataset = str(ds)

    # variants
    raw_variants = d.get("variants", {})
    if isinstance(raw_variants, dict):
        tc.variants = {str(k): str(v).strip() for k, v in raw_variants.items()}

    # expect
    raw_expect = d.get("expect", {})
    if isinstance(raw_expect, dict):
        tc.expect = _parse_expect_dict(raw_expect)

    return tc


def _parse_expect_dict(d: dict) -> Expect:
    e = Expect()
    e.ordered = bool(d.get("ordered", False))
    e.empty = bool(d.get("empty", False))

    count = d.get("count")
    if count is not None:
        e.count = int(count)

    error = d.get("error")
    if error is not None:
        e.error = str(error)

    hash_val = d.get("hash")
    if hash_val is not None:
        e.hash = str(hash_val)

    precision = d.get("precision")
    if precision is not None:
        e.precision = int(precision)

    e.columns = _as_string_list(d.get("columns", []))

    # rows: list of lists, each element becomes a string for comparison
    raw_rows = d.get("rows", [])
    for raw_row in raw_rows:
        if isinstance(raw_row, list):
            e.rows.append([_value_to_string(v) for v in raw_row])
        else:
            # Single-column shorthand
            e.rows.append([_value_to_string(raw_row)])

    return e


# ---------------------------------------------------------------------------
# Line-based fallback parser
# ---------------------------------------------------------------------------


def _parse_line_based(content: str, path: Path) -> GtestFile:
    """Minimal line-based parser for when PyYAML is not installed.

    This handles the subset of YAML used by .gtest files:
    top-level ``meta:`` / ``tests:`` blocks, inline ``[a, b]`` lists,
    block scalars (``|``), and ``- name:`` list items.
    """
    lines = content.splitlines()
    idx = _skip_blank_and_comments(lines, 0)

    # Parse meta block
    meta, idx = _lb_parse_meta(lines, idx)
    idx = _skip_blank_and_comments(lines, idx)

    # Parse tests block
    tests, idx = _lb_parse_tests(lines, idx)

    return GtestFile(meta=meta, tests=tests)


def _lb_parse_meta(lines: list[str], idx: int) -> tuple:
    meta = Meta()
    if idx < len(lines) and lines[idx].strip() == "meta:":
        idx += 1
    while idx < len(lines):
        line = lines[idx]
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
            continue
        if not line[0].isspace():
            break
        key, value = _lb_parse_kv(trimmed)
        if key == "language":
            meta.language = value
        elif key == "model":
            meta.model = value
        elif key == "section":
            meta.section = _unquote(value)
        elif key == "title":
            meta.title = value
        elif key == "dataset":
            meta.dataset = value
        elif key == "requires":
            meta.requires = _lb_parse_yaml_list(value)
        elif key == "tags":
            meta.tags = _lb_parse_yaml_list(value)
        elif key == "iso":
            meta.iso = _lb_parse_yaml_list(value)
        idx += 1
    return meta, idx


def _lb_parse_tests(lines: list[str], idx: int) -> tuple:
    tests: list[TestCase] = []
    if idx < len(lines) and lines[idx].strip() == "tests:":
        idx += 1
    while idx < len(lines):
        idx = _skip_blank_and_comments(lines, idx)
        if idx >= len(lines):
            break
        trimmed = lines[idx].strip()
        if trimmed.startswith("- name:"):
            tc, idx = _lb_parse_single_test(lines, idx)
            tests.append(tc)
        else:
            break
    return tests, idx


def _lb_parse_single_test(lines: list[str], idx: int) -> tuple:
    tc = TestCase()
    first = lines[idx].strip()
    # "- name: foo"
    _, name_val = _lb_parse_kv(first[2:])  # strip "- "
    tc.name = _unquote(name_val)
    idx += 1

    while idx < len(lines):
        trimmed = lines[idx].strip()
        if trimmed.startswith("#"):
            idx += 1
            continue
        if trimmed.startswith("- name:"):
            break
        if not trimmed:
            idx += 1
            continue

        key, value = _lb_parse_kv(trimmed)
        if key == "query":
            if value == "|":
                block, idx = _lb_parse_block_scalar(lines, idx)
                tc.query = block
            else:
                tc.query = _unquote(value)
                idx += 1
        elif key == "skip":
            tc.skip = _unquote(value)
            idx += 1
        elif key == "setup":
            idx += 1
            tc.setup, idx = _lb_parse_string_list(lines, idx)
        elif key == "statements":
            idx += 1
            tc.statements, idx = _lb_parse_string_list(lines, idx)
        elif key == "tags":
            tc.tags = _lb_parse_yaml_list(value)
            idx += 1
        elif key == "requires":
            tc.requires = _lb_parse_yaml_list(value)
            idx += 1
        elif key == "iso":
            tc.iso = _lb_parse_yaml_list(value)
            idx += 1
        elif key == "params":
            idx += 1
            tc.params, idx = _lb_parse_params(lines, idx)
        elif key == "expect":
            idx += 1
            tc.expect, idx = _lb_parse_expect(lines, idx)
        elif key == "variants":
            idx += 1
            tc.variants, idx = _lb_parse_variants(lines, idx)
        else:
            idx += 1

    return tc, idx


def _lb_parse_expect(lines: list[str], idx: int) -> tuple:
    e = Expect()
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
            continue
        if trimmed.startswith("- name:"):
            break
        if not lines[idx][0].isspace():
            break

        key, value = _lb_parse_kv(trimmed)
        if key == "ordered":
            e.ordered = value == "true"
            idx += 1
        elif key == "count":
            e.count = int(value)
            idx += 1
        elif key == "empty":
            e.empty = value == "true"
            idx += 1
        elif key == "error":
            e.error = _unquote(value)
            idx += 1
        elif key == "hash":
            e.hash = _unquote(value)
            idx += 1
        elif key == "precision":
            e.precision = int(value)
            idx += 1
        elif key == "columns":
            e.columns = _lb_parse_yaml_list(value)
            idx += 1
        elif key == "rows":
            idx += 1
            e.rows, idx = _lb_parse_rows(lines, idx)
        else:
            break
    return e, idx


def _lb_parse_rows(lines: list[str], idx: int) -> tuple:
    rows: list[list[str]] = []
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
            continue
        if trimmed.startswith("- ["):
            inner = trimmed[2:]  # strip "- "
            values = _lb_parse_inline_list(inner)
            rows.append(values)
            idx += 1
        else:
            break
    return rows, idx


def _lb_parse_string_list(lines: list[str], idx: int) -> tuple:
    items: list[str] = []
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
            continue
        if trimmed.startswith("- "):
            value = trimmed[2:]
            if value == "|":
                block, idx = _lb_parse_block_scalar(lines, idx)
                items.append(block)
            else:
                items.append(_unquote(value))
                idx += 1
        else:
            break
    return items, idx


def _lb_parse_params(lines: list[str], idx: int) -> tuple:
    """Params entries sit at indent 6 or deeper; each value keeps its quotes."""
    params: dict[str, str] = {}
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        if indent >= 6:
            key, value = _lb_parse_kv(trimmed)
            params[key] = value
            idx += 1
        else:
            break
    return params, idx


def _lb_parse_variants(lines: list[str], idx: int) -> tuple:
    variants: dict[str, str] = {}
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        if indent >= 6:
            key, value = _lb_parse_kv(trimmed)
            if value == "|":
                block, idx = _lb_parse_block_scalar(lines, idx)
                variants[key] = block
            else:
                variants[key] = _unquote(value)
                idx += 1
        else:
            break
    return variants, idx


def _lb_parse_block_scalar(lines: list[str], idx: int) -> tuple:
    """Parse a YAML block scalar (line ending with ``|``)."""
    idx += 1  # skip the ``|`` line
    if idx >= len(lines):
        return "", idx
    block_indent = len(lines[idx]) - len(lines[idx].lstrip())
    parts: list[str] = []
    while idx < len(lines):
        line = lines[idx]
        trimmed = line.strip()
        if not trimmed:
            parts.append("")
            idx += 1
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent < block_indent:
            break
        parts.append(line[block_indent:])
        idx += 1
    text = "\n".join(parts).rstrip()
    return text, idx


# ---------------------------------------------------------------------------
# Parameter values
# ---------------------------------------------------------------------------

# The number syntax of the Rust reference runner (crates/grafeo-spec-tests/
# build.rs): decimal only, so ``0x1F``, ``1_000``, ``inf`` and ``NaN`` are not
# numbers. ``[0-9]``, not ``\d``, which also matches other scripts' digits.
_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def param_value(text: str) -> object:
    """The value of a parameter written as ``text`` in a .gtest file.

    The rule of the Rust reference runner, which every runner follows: a
    quoted value is a string, whatever it reads like; a bare value starting
    with ``[`` or ``{`` is JSON; a bare decimal integer that fits i64 is an
    int and any other bare decimal number a float; a bare ``true`` or
    ``false`` is a bool; any other bare value is a string.
    """
    text = text.strip()
    if _is_quoted(text):
        return _unquote(text)
    if text.startswith(("[", "{")):
        return json.loads(
            text,
            parse_int=_json_integer,
            parse_float=_finite_float,
            parse_constant=_reject_json_constant,
        )
    if _DECIMAL_INTEGER.fullmatch(text) and _I64_MIN <= int(text) <= _I64_MAX:
        return int(text)
    if _DECIMAL_NUMBER.fullmatch(text):
        return _finite_float(text)
    if text in ("true", "false"):
        return text == "true"
    return text


def _json_integer(text: str) -> int | float:
    """A JSON integer, by the rule of a bare one: an int when it fits i64, a
    float when it does not (``json.loads`` alone keeps any size)."""
    number = int(text)
    if _I64_MIN <= number <= _I64_MAX:
        return number
    return _finite_float(text)


def _finite_float(text: str) -> float:
    """``text`` as a float; past the f64 range (``1e400``, which ``float``
    reads as infinity) it is an error, as in every runner."""
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"parameter {text!r} is out of the f64 range")
    return number


def _reject_json_constant(name: str) -> object:
    """``json.loads`` reads ``NaN`` and ``Infinity``; JSON (and Rust) do not."""
    raise ValueError(f"{name} is not a JSON value")


def coerce_params(raw_params: dict[str, str]) -> dict[str, object] | None:
    """Type each parameter by ``param_value``; None when there are none."""
    if not raw_params:
        return None
    return {key: param_value(text) for key, text in raw_params.items()}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _skip_blank_and_comments(lines: list[str], idx: int) -> int:
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("#"):
            idx += 1
        else:
            break
    return idx


def _lb_parse_kv(s: str) -> tuple:
    """Split ``key: value`` respecting quotes."""
    in_single = False
    in_double = False
    for i, c in enumerate(s):
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == ":" and not in_single and not in_double:
            key = s[:i].strip()
            value = s[i + 1 :].strip()
            if key:
                return key, value
    return s.strip(), ""


def _is_quoted(s: str) -> bool:
    """Whether ``s`` is written in single or double quotes."""
    s = s.strip()
    return len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
    )


def _unquote(s: str) -> str:
    """Strip surrounding quotes and unescape YAML-level escapes only.

    Do NOT process ``\\n`` or ``\\t`` here: those are GQL string escapes
    that the engine's parser handles via ``unescape_string()``.
    """
    s = s.strip()
    if _is_quoted(s):
        inner = s[1:-1]
        # Use a sentinel to avoid order-dependent replacement issues
        return (
            inner.replace("\\\\", "\x00")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\x00", "\\")
        )
    return s


def _lb_parse_yaml_list(s: str) -> list[str]:
    s = s.strip()
    if s == "[]" or not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        return [_unquote(v.strip()) for v in inner.split(",") if v.strip()]
    return [_unquote(s)]


def _lb_parse_inline_list(s: str) -> list[str]:
    """Parse ``[a, b, c]`` respecting nested brackets and quotes."""
    s = s.strip()
    if not s.startswith("[") or not s.endswith("]"):
        return [_unquote(s)]
    inner = s[1:-1]

    items: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False

    for c in inner:
        if c == "'" and not in_double and depth == 0:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single and depth == 0:
            in_double = not in_double
            current.append(c)
        elif c in "[{" and not in_single and not in_double:
            depth += 1
            current.append(c)
        elif c in "]}" and not in_single and not in_double:
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0 and not in_single and not in_double:
            items.append(_unquote("".join(current).strip()))
            current = []
        else:
            current.append(c)

    last = "".join(current).strip()
    if last:
        items.append(_unquote(last))

    return items


def _as_string_list(val) -> list[str]:
    """Coerce a YAML value to a list of strings."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [str(v).strip() if v is not None else "" for v in val]
    return [str(val)]


def _value_to_string(val) -> str:
    """Convert a YAML-parsed Python value to the canonical string the Rust
    runner would produce.  This is the Python equivalent of
    ``value_to_string`` in ``grafeo-spec-tests/src/lib.rs``.
    """
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if val == float("inf"):
            return "Infinity"
        if val == float("-inf"):
            return "-Infinity"
        # Rust's Display for f64 drops ".0" for whole numbers.
        if val == int(val) and abs(val) < 2**53:
            return str(int(val))
        return str(val)
    if isinstance(val, list):
        inner = ", ".join(_value_to_string(v) for v in val)
        return f"[{inner}]"
    if isinstance(val, dict):
        entries = sorted(f"{k}: {_value_to_string(v)}" for k, v in val.items())
        return "{" + ", ".join(entries) + "}"
    return str(val)
