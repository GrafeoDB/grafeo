"""Leaving soft-deleted nodes out of text search from Python (#742): `filters={"deleted":
{"$ne": True}}` keeps a node that has no `deleted` flag, and a null filter value is refused
with an invalid-argument error pointing to `$ne`.
"""

import pytest

grafeo = pytest.importorskip("grafeo")


def documents():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (:Doc {owner: 'Alix', text: 'canals of Amsterdam', deleted: false}), "
        "(:Doc {owner: 'Gus', text: 'canals and bridges', deleted: true}), "
        "(:Doc {owner: 'Vincent', text: 'canals at night'}), "
        "(:Doc {owner: 'Jules', text: 'Amsterdam canals in winter', deleted: false})"
    )
    db.create_text_index("Doc", "text")
    return db


def owners(db, hits):
    found = set()
    for node_id, _ in hits:
        rows = db.execute(f"MATCH (d:Doc) WHERE id(d) = {node_id} RETURN d.owner AS o")
        found.add(next(iter(rows))["o"])
    return found


@pytest.mark.requires("text-index")
def test_ne_keeps_a_node_without_the_flag():
    db = documents()
    hits = db.text_search("Doc", "text", "canals", 10, filters={"deleted": {"$ne": True}})
    assert owners(db, hits) == {"Alix", "Vincent", "Jules"}
    hits = db.text_search("Doc", "text", "canals", 10, filters={"deleted": False})
    assert owners(db, hits) == {"Alix", "Jules"}, "equality needs the flag"


@pytest.mark.requires("text-index")
@pytest.mark.parametrize("filters", [{"deleted": None}, {"deleted": {"$ne": None}}])
def test_a_null_filter_value_is_refused(filters):
    with pytest.raises((ValueError, grafeo.GrafeoError)) as raised:
        documents().text_search("Doc", "text", "canals", 10, filters=filters)
    assert "deleted" in str(raised.value), str(raised.value)
    assert "$ne" in str(raised.value), str(raised.value)
