"""GQL text_match on a database file (#308).

text_match matched nothing on a file database, so a WHERE kept no row. It
finds the document through the text index, after a reopen and read-only too,
and without a text index on the property it is an error that says to create
one.
"""

import pytest
from grafeo import GrafeoDB, GrafeoError

QUERY = "MATCH (d:Doc) WHERE text_match(d.body, 'canals') RETURN d.title AS t"


def titles(db):
    return [row["t"] for row in db.execute(QUERY)]


def test_text_match_on_a_file_database(tmp_path):
    path = str(tmp_path / "docs.grafeo")
    db = GrafeoDB(path)
    db.execute("INSERT (:Doc {title: 'Canals', body: 'boats on the canals of Amsterdam'})")
    db.execute("INSERT (:Doc {title: 'Bridges', body: 'the bridges of Prague'})")
    db.execute("CREATE INDEX doc_body FOR (d:Doc) ON (d.body) USING TEXT")
    assert titles(db) == ["Canals"]
    db.close()

    db = GrafeoDB(path)
    assert titles(db) == ["Canals"], "after a reopen"
    db.close()
    db = GrafeoDB.open_read_only(path)
    assert titles(db) == ["Canals"], "read-only"
    db.close()


def test_text_match_without_a_text_index_says_to_create_one(tmp_path):
    db = GrafeoDB(str(tmp_path / "docs.grafeo"))
    db.execute("INSERT (:Doc {title: 'Canals', body: 'boats on the canals of Amsterdam'})")
    with pytest.raises(GrafeoError, match="needs a text index on 'body'"):
        db.execute(QUERY)
    db.close()
