"""A native object a binding owns stays alive for as long as a native call uses it.

A garbage collector may finalize a wrapper whose last use is the call in progress, and its
finalizer frees the native object under the call. Each binding with finalizers guards this its own
way, and these scans hold every wrapper to it (Go's own scan is `keepalive_test.go` in its package):

- C#: an owned object reaches P/Invoke as its SafeHandle type, which the marshaller references and
  releases around the call; a raw `nint` from `DangerousGetHandle()` does not keep it.
- Dart: a class that attaches a `NativeFinalizer` implements `Finalizable`, which the compiler keeps
  alive to the end of the enclosing scope.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSHARP = ROOT / "crates/bindings/csharp/src/Grafeo"
DART = ROOT / "crates/bindings/dart/lib"

# The P/Invoke parameter names of the objects a wrapper owns through a SafeHandle (and its critical
# finalizer), and that SafeHandle type. A result, node or edge is a local the same method frees, and
# the stream has no finalizer: no collector frees those under a call. The release imports
# (`ReleaseMethods`, in Native/Handles.cs) take the raw handle `ReleaseHandle` holds, so only
# NativeMethods.cs is read.
OWNED = {
    "db": "DatabaseHandle",
    "graph": "GraphHandle",
    "tx": "TransactionHandle",
    "view": "AsOfHandle",
}


def native_parameters() -> list[tuple[str, int, str, str]]:
    """`(function, line, type, name)` for every parameter of a `grafeo_*` P/Invoke declaration."""
    source = (CSHARP / "Native/NativeMethods.cs").read_text(encoding="utf-8")
    found = []
    for match in re.finditer(r"\b(grafeo_\w+)\s*\(([^)]*)\)\s*;", source):
        line = source.count("\n", 0, match.start()) + 1
        for parameter in match.group(2).split(","):
            parts = re.sub(r"\[[^\]]*\]", "", parameter).split()
            if len(parts) >= 2:
                found.append((match.group(1), line, parts[-2], parts[-1]))
    return found


def test_csharp_owned_objects_reach_native_calls_as_safe_handles() -> None:
    parameters = native_parameters()
    assert len(parameters) > 100, "the scan reads the P/Invoke declarations"
    wrong = [
        f"NativeMethods.cs:{line} {function}: `{kind} {name}`, want {OWNED[name]}"
        for function, line, kind, name in parameters
        if name in OWNED and kind != OWNED[name]
    ]
    assert not wrong, "\n".join(wrong)


def test_csharp_takes_no_raw_handle_outside_the_handle_types() -> None:
    wrong = []
    for path in sorted(CSHARP.rglob("*.cs")):
        if path.name == "Handles.cs":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "DangerousGetHandle()" in line:
                wrong.append(
                    f"{path.relative_to(ROOT).as_posix()}:{number}: {line.strip()}"
                )
    assert not wrong, "\n".join(wrong)


def test_every_dart_class_with_a_native_finalizer_is_finalizable() -> None:
    checked = 0
    wrong = []
    for path in sorted(DART.rglob("*.dart")):
        source = path.read_text(encoding="utf-8")
        # Each class body, up to the next top-level class.
        for match in re.finditer(
            r"(?m)^(?:final |abstract |base |sealed )*class (\w+)([^{]*)\{", source
        ):
            end = re.search(
                r"(?m)^(?:final |abstract |base |sealed )*class \w+",
                source[match.end() :],
            )
            body = (
                source[match.end() : match.end() + end.start()]
                if end
                else source[match.end() :]
            )
            # Code only: a doc comment may name `NativeFinalizer` before the next class.
            code = re.sub(r"//.*", "", body)
            if "NativeFinalizer" not in code and ".attach(" not in code:
                continue
            checked += 1
            if not re.search(
                r"\b(implements|with)\b[^{]*\bFinalizable\b", match.group(2)
            ):
                wrong.append(
                    f"{path.relative_to(ROOT).as_posix()}: class {match.group(1)}"
                )
    assert checked >= 5, f"the scan found {checked} classes with a native finalizer"
    assert not wrong, "\n".join(wrong)
