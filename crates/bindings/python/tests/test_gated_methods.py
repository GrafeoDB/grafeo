"""A method the binding compiles only with a feature is there exactly when
grafeo.features() reports that feature.

The gated methods are read from the binding's own source: every method of a
`#[pymethods]` block under a `#[cfg(...)]`, by its Python name. So a method
gated later is checked without a change here. A custom profile (lpg, rdf, ai,
analytics, edge) builds the binding without some features; its methods for
them must be absent, not present and failing.
"""

import re
from pathlib import Path

import grafeo
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def classes():
    """The Python class of each name a gated method can sit on."""
    db = grafeo.GrafeoDB()
    native = grafeo.grafeo
    found = {
        "GrafeoDB": grafeo.GrafeoDB,
        "Transaction": type(db.begin_transaction()),
        "QueryResult": type(db.execute("RETURN 1 AS v")),
    }
    for name in ["GraphHandle", "Node", "Edge", "ResultStream", "AsOfView", "Value", "Session"]:
        found[name] = getattr(grafeo, name, None) or getattr(native, name)
    return found


def cfg_holds(expression, features):
    """Whether a `cfg(...)` expression of feature terms holds for `features`."""
    expression = expression.strip()
    term = re.fullmatch(r'feature\s*=\s*"([^"]+)"', expression)
    if term:
        return term.group(1) in features
    call = re.fullmatch(r"(all|any|not)\s*\((.*)\)", expression, re.S)
    if not call:
        raise ValueError(f"a cfg the test cannot read: {expression}")
    parts, depth, start = [], 0, 0
    inner = call.group(2)
    for index, char in enumerate(inner):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(inner[start:index])
            start = index + 1
    parts.append(inner[start:])
    values = [cfg_holds(part, features) for part in parts if part.strip()]
    if call.group(1) == "not":
        return not values[0]
    return all(values) if call.group(1) == "all" else any(values)


def cfg_features(expression):
    """The feature names a `cfg(...)` expression reads."""
    return set(re.findall(r'feature\s*=\s*"([^"]+)"', expression))


def python_names(source):
    """Rust struct name to Python class name, from `#[pyclass(name = ...)]`."""
    names = {}
    declaration = r"#\[pyclass(?:\(([^]]*)\))?\]\s*(?:#\[[^]]*\]\s*)*pub struct (\w+)"
    for match in re.finditer(declaration, source):
        rename = re.search(r'name\s*=\s*"(\w+)"', match.group(1) or "")
        names[match.group(2)] = rename.group(1) if rename else match.group(2)
    return names


def gated_methods(source, names):
    """`{(class, method): [cfg or None, ...]}` for the methods of the `#[pymethods]` blocks that
    have a cfg in some variant: one entry per definition, None for one without a cfg."""
    methods = {}
    for block in re.finditer(r"(?m)^#\[pymethods\]\s*\nimpl (\w+) \{\n(.*?)^\}", source, re.S):
        python_class = names.get(block.group(1), block.group(1))
        pending = []
        for line in block.group(2).splitlines():
            if line.startswith("    }"):
                pending = []
                continue
            if not line.startswith("    ") or line.startswith("     ") and not pending:
                continue
            stripped = line.strip()
            definition = re.match(r"(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?fn\s+(\w+)", stripped)
            if definition and line.startswith("    ") and not line.startswith("     "):
                attributes = "\n".join(pending)
                pending = []
                if "#[setter" in attributes:
                    continue
                name = definition.group(1)
                rename = re.search(r'#\[pyo3\([^]]*?\bname\s*=\s*"(\w+)"', attributes, re.S)
                getter = re.search(r"#\[getter(?:\((\w+)\))?\]", attributes)
                if rename:
                    name = rename.group(1)
                elif getter:
                    name = getter.group(1) or name.removeprefix("get_")
                cfgs = re.findall(r"#\[cfg\((.*?)\)\]\s*$", attributes, re.M)
                methods.setdefault((python_class, name), []).append(
                    " , ".join(cfgs) if cfgs else None
                )
            elif stripped.startswith(("#", "///", "//")) or pending:
                pending.append(stripped)
    return {key: cfgs for key, cfgs in methods.items() if any(cfgs)}


def binding_methods():
    texts = [path.read_text(encoding="utf-8") for path in sorted(SRC.rglob("*.rs"))]
    names = {}
    for text in texts:
        names.update(python_names(text))
    methods = {}
    for text in texts:
        for key, cfgs in gated_methods(text, names).items():
            methods.setdefault(key, []).extend(cfgs)
    return methods


def reportable():
    """The names features() can report: the pairs of the shared `reported_features!` macro in
    bindings-common, which every binding's features() calls."""
    source = (SRC.parent.parent / "common" / "src" / "lib.rs").read_text(encoding="utf-8")
    body = source[source.index("macro_rules! reported_features") :]
    body = body[: body.index(".into_iter()")]
    return set(re.findall(r'\("([a-z-]+)", cfg!\(feature = "\1"\)\)', body))


SAMPLE = """
#[pyclass(name = "GrafeoDB")]
pub struct PyGrafeoDB {
    inner: u8,
}

#[pymethods]
impl PyGrafeoDB {
    /// Always there.
    fn execute(&self) {}

    #[cfg(feature = "cypher")]
    #[pyo3(signature = (
        query,
        params=None
    ))]
    fn execute_cypher(&self) {
        let text = format!("{}", 1);
    }

    #[cfg(all(feature = "embed", feature = "vector-index"))]
    #[pyo3(name = "vector_search_text")]
    fn search_text(&self) {}

    #[cfg(feature = "metrics")]
    #[getter]
    fn get_metrics(&self) {}

    #[cfg(feature = "algos")]
    fn both(&self) {}

    #[cfg(not(feature = "algos"))]
    fn both(&self) {}
}

impl PyGrafeoDB {
    #[cfg(feature = "arrow-export")]
    fn nodes_ipc_bytes(&self) {}
}
"""


def test_the_reader_finds_each_gated_method_by_its_python_name():
    names = python_names(SAMPLE)
    assert names == {"PyGrafeoDB": "GrafeoDB"}
    found = gated_methods(SAMPLE, names)
    assert found == {
        ("GrafeoDB", "execute_cypher"): ['feature = "cypher"'],
        ("GrafeoDB", "vector_search_text"): ['all(feature = "embed", feature = "vector-index")'],
        ("GrafeoDB", "metrics"): ['feature = "metrics"'],
        ("GrafeoDB", "both"): ['feature = "algos"', 'not(feature = "algos")'],
    }
    assert cfg_holds('all(feature = "embed", feature = "vector-index")', {"embed"}) is False
    assert cfg_holds('any(feature = "embed", feature = "vector-index")', {"embed"}) is True
    assert cfg_holds('not(feature = "algos")', set()) is True


def test_every_gated_method_names_features_the_build_reports():
    methods = binding_methods()
    assert len(methods) >= 40, sorted(methods)
    known = reportable()
    unknown = {
        f"{cls}.{name}": sorted(cfg_features(cfg) - known)
        for (cls, name), cfgs in methods.items()
        for cfg in cfgs
        if cfg and cfg_features(cfg) - known
    }
    assert not unknown, f"gated on a feature features() cannot report: {unknown}"


def test_a_gated_method_is_there_exactly_when_its_features_are():
    features = set(grafeo.features())
    found = classes()
    wrong = []
    for (cls, name), cfgs in sorted(binding_methods().items()):
        if cls not in found:
            pytest.fail(f"{cls} holds a gated method: add how to reach the class to classes()")
        expected = any(cfg is None or cfg_holds(cfg, features) for cfg in cfgs)
        if hasattr(found[cls], name) != expected:
            wanted = "there" if expected else "absent"
            wrong.append(f"{cls}.{name} should be {wanted}: {cfgs}")
    assert not wrong, f"features {sorted(features)}: {wrong}"
