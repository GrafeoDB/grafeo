"""A refused permission raises GrafeoPermissionError, a GrafeoError, with GQLSTATUS 42501 and its
own code: GRAFEO-Q007 for a write the session may not make, GRAFEO-Q008 for external access the
session does not have. A mistake in the query keeps its own error.
"""

import pytest

grafeo = pytest.importorskip("grafeo")


def test_a_write_on_a_read_only_session_is_a_permission_error():
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoPermissionError) as raised:
        db.session(read_only=True).execute("INSERT (:Denied)")
    assert isinstance(raised.value, grafeo.GrafeoError)
    assert raised.value.error_code == "GRAFEO-Q007", str(raised.value)
    assert raised.value.gql_status == "42501", str(raised.value)


def test_external_access_refused_is_a_permission_error(tmp_path):
    db = grafeo.GrafeoDB()
    path = (tmp_path / "people.csv").as_posix()
    with pytest.raises(grafeo.GrafeoPermissionError) as raised:
        db.session(external_access=False).execute(
            f"CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['P']}})"
        )
    assert raised.value.error_code == "GRAFEO-Q008", str(raised.value)
    assert raised.value.gql_status == "42501", str(raised.value)


def test_an_unknown_variable_is_no_permission_error():
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute("RETURN nobody.name")
    assert not isinstance(raised.value, grafeo.GrafeoPermissionError)
    assert raised.value.error_code == "GRAFEO-Q002", str(raised.value)
    assert raised.value.gql_status == "42002", str(raised.value)
