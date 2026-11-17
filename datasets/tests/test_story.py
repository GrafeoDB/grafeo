"""Story checks: each one is a GQL query with the answer the source gives."""

import json

import pytest

from conftest import load_lines


@pytest.fixture
def loaded(db, derived):
    load_lines(db, derived / "silicon-valley.gql", "execute")
    return db


def column(db, query, name):
    return [row[name] for row in db.execute(query)]


def test_middle_out_weissman_score(loaded):
    scores = column(
        loaded, "MATCH (n {key: 'middle-out'}) RETURN n.weissman_score AS v", "v"
    )
    assert scores == [5.2]


def test_pied_piper_moved_from_palo_alto_to_mountain_view(loaded):
    early = column(
        loaded,
        "MATCH (c:Company {key: 'pied-piper'})-[b:BASED_IN]->(:Place)-[:LOCATED_IN]->(city:City) "
        "WHERE b.from < DATE '2014-09-01' RETURN city.key AS v",
        "v",
    )
    assert early == ["palo-alto"]
    late = column(
        loaded,
        "MATCH (c:Company {key: 'pied-piper'})-[b:BASED_IN]->(:Place)-[:LOCATED_IN]->(city:City) "
        "WHERE b.from = DATE '2014-09-01' AND b.to IS NULL RETURN city.key AS v",
        "v",
    )
    assert late == ["mountain-view"]


def test_gilfoyle_built_anton(loaded):
    rows = column(
        loaded,
        "MATCH (:Person {key: 'bertram-gilfoyle'})-[:BUILT]->(n {key: 'anton'}) RETURN n.name AS v",
        "v",
    )
    assert rows == ["Anton"]


def test_son_of_anton_was_trained_on_anton(loaded):
    rows = column(
        loaded,
        "MATCH (:AI {key: 'son-of-anton'})-[:TRAINED_ON]->(n:AI) RETURN n.key AS v",
        "v",
    )
    assert rows == ["anton"]


def test_hooli_ceos_in_order(loaded):
    rows = list(
        loaded.execute(
            "MATCH (p:Person)-[w:WORKS_AT {role: 'CEO'}]->(:Company {key: 'hooli'}) "
            "RETURN p.key AS who, w.from AS since, w.to AS until "
            "ORDER BY coalesce(w.from, DATE '0001-01-01')"
        )
    )
    assert [r["who"] for r in rows] == ["gavin-belson", "jack-barker", "gavin-belson"]
    assert [(str(r["since"]), str(r["until"])) for r in rows] == [
        ("None", "2017-06-01"),
        ("2017-06-01", "2018-04-01"),
        ("2018-04-01", "None"),
    ]


def test_jared_born_as_donald_dunn(loaded):
    assert column(
        loaded, "MATCH (n {key: 'jared-dunn'}) RETURN n.born_as AS v", "v"
    ) == ["Donald Dunn"]


def test_ron_works_at_pied_piper(loaded):
    rows = column(
        loaded,
        "MATCH (:Person {key: 'ron-laflamme'})-[:WORKS_AT]->(c:Company) RETURN c.key AS v",
        "v",
    )
    assert "pied-piper" in rows


def first_of_month(value):
    return value + "-01" if len(value) == 7 else value


def lit(value):
    """A GQL literal: strings quoted and escaped, numbers and booleans bare."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = value.replace(chr(92), chr(92) * 2).replace("'", chr(92) + "'")
    return "'" + escaped + "'"


def edge_matches(db, fact):
    """True when an edge of the fact's shape carries every property, dates compared by their first day."""
    conditions = []
    for name, value in fact["props"].items():
        if name in ("from", "to", "date"):
            conditions.append(f"e.{name} = DATE '{first_of_month(value)}'")
            conditions.append(
                f"e.{name}_precision = 'month'"
                if len(value) == 7
                else f"e.{name}_precision IS NULL"
            )
        else:
            conditions.append(f"e.{name} = {lit(value)}")
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    query = (
        f"MATCH (a {{key: {lit(fact['source'])}}})-[e:{fact['type']}]->(b {{key: {lit(fact['target'])}}}){where} "
        "RETURN count(e) AS c"
    )
    return list(db.execute(query))[0]["c"] > 0


def test_replay_facts_hold(loaded, fixtures):
    facts = json.loads((fixtures / "replay-facts.json").read_text(encoding="utf-8"))
    assert facts
    for fact in facts:
        assert edge_matches(loaded, fact), f"no edge with these properties: {fact}"
