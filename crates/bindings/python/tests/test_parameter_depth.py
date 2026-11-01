"""A parameter nested deeper than a property value may be is refused.

A list or dict parameter nested past 128 levels (the depth a stored value may
have) raises ValueError naming the limit, as the binding's other value
conversions do (or GrafeoError, should the engine refuse it), and the
process lives on: no stack
overflow while the binding converts it or the engine reads it, also at depth
100,000. The deep cases run in a subprocess, since a stack overflow ends the
interpreter.
"""

import subprocess
import sys
import textwrap

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

LIMIT = 128

CHILD = textwrap.dedent(
    """
    import sys
    import grafeo

    shape, depth, query = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    value = 1
    for _ in range(depth):
        value = [value] if shape == "list" else {"a": value}
    db = grafeo.GrafeoDB()
    try:
        db.execute(query, {"p": value})
    except (ValueError, grafeo.GrafeoError) as error:
        print("REFUSED", error)
        assert db.execute("RETURN 1 AS one").scalar() == 1
        sys.exit(19)
    print("ACCEPTED")
    sys.exit(23)
    """
)


def nested(shape, depth):
    value = 1
    for _ in range(depth):
        value = [value] if shape == "list" else {"a": value}
    return value


@pytest.mark.parametrize("query", ["RETURN $p AS p", "INSERT (:N {p: $p})"])
@pytest.mark.parametrize("shape", ["list", "dict"])
@pytest.mark.parametrize("depth", [LIMIT + 1, 100_000])
def test_a_parameter_nested_past_the_limit_is_refused(depth, shape, query):
    child = subprocess.run(
        [sys.executable, "-c", CHILD, shape, str(depth), query],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = child.stdout + child.stderr
    want = "want the error and a live process"
    assert child.returncode == 19, f"exit {child.returncode}, {want}:\n{output}"
    assert str(LIMIT) in output, f"the error names the limit:\n{output}"


@pytest.mark.parametrize("shape", ["list", "dict"])
def test_a_parameter_at_the_limit_works(shape):
    db = grafeo.GrafeoDB()
    value = nested(shape, LIMIT)
    assert db.execute("RETURN $p = $p AS same", {"p": value}).scalar() is True
    db.execute("INSERT (:N {p: $p})", {"p": value})
    assert db.execute("MATCH (n:N) RETURN count(n)").scalar() == 1
