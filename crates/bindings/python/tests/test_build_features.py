"""grafeo.features() and grafeo.build_info() describe the build."""

import os
import re

import grafeo
import pytest

QUERY_LANGUAGES = {"gql", "cypher", "sparql", "gremlin", "graphql", "sql-pgq"}

# Feature name -> a GrafeoDB attribute that exists only when it is compiled in.
GATED_ATTRIBUTES = {
    "cypher": "execute_cypher",
    "sparql": "execute_sparql",
    "gremlin": "execute_gremlin",
    "graphql": "execute_graphql",
    "sql-pgq": "execute_sql",
    "shacl": "validate_shacl",
    "algos": "algorithms",
}


def test_build_reports_its_features():
    features = grafeo.features()
    assert isinstance(features, list) and "gql" in features
    assert all(isinstance(name, str) for name in features)
    assert len(features) == len(set(features))


def test_features_match_the_api_of_this_build():
    features = grafeo.features()
    db = grafeo.GrafeoDB()
    for feature, attribute in GATED_ATTRIBUTES.items():
        assert (feature in features) == hasattr(db, attribute), feature


def test_features_are_exported_from_the_package():
    assert {"features", "build_info"} <= set(grafeo.__all__)


def test_build_info_describes_this_build():
    info = grafeo.build_info()
    assert set(info) == {"version", "commit", "dirty", "features", "profile"}
    assert info["version"] == grafeo.__version__
    assert info["features"] == grafeo.features()
    assert info["profile"] in ("release", "debug")
    if info["commit"] is None:
        # Built outside a git checkout (for example from an sdist).
        assert info["dirty"] is None
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", info["commit"]), info["commit"]
        assert isinstance(info["dirty"], bool)


@pytest.mark.skipif(
    os.environ.get("GRAFEO_RELEASE_BUILD") != "1",
    reason="checks a release-equivalent build: set GRAFEO_RELEASE_BUILD=1",
)
def test_release_build_has_every_query_language():
    assert QUERY_LANGUAGES <= set(grafeo.features())
