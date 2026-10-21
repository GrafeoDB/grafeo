"""`import_csv` and `import_jsonl` keep every column and key under its own name.

The binding built its own `LOAD DATA` statement with sanitized names, so a
header with a space (`home city`), a byte order mark, a quoted header holding
the delimiter, or a JSONL key that first appears on a later line lost its
values without an error. Both calls now run the engine's `import_nodes`.
"""

import grafeo
import pytest


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def nodes(db, label):
    label = label.replace("`", "``")
    rows = db.execute(f"MATCH (n:`{label}`) RETURN properties(n) AS p")
    return sorted((row["p"] for row in rows), key=lambda p: sorted(p.items(), key=str))


def test_a_header_with_a_space_keeps_its_values(tmp_path):
    path = write(tmp_path, "people.csv", "name,home city\nAlix,Amsterdam\nGus,Berlin\n")
    db = grafeo.GrafeoDB()
    assert db.import_csv(path, label="Person") == 2
    assert nodes(db, "Person") == [
        {"name": "Alix", "home city": "Amsterdam"},
        {"name": "Gus", "home city": "Berlin"},
    ]


def test_a_byte_order_mark_and_a_quoted_header(tmp_path):
    path = write(tmp_path, "excel.csv", '﻿id,"Amsterdam, NL"\n3,yes\n')
    db = grafeo.GrafeoDB()
    assert db.import_csv(path) == 1
    assert nodes(db, "Row") == [{"id": "3", "Amsterdam, NL": "yes"}]


def test_a_csv_without_headers_numbers_its_columns(tmp_path):
    path = write(tmp_path, "raw.csv", "Alix,Amsterdam\nGus,Berlin,3\n")
    db = grafeo.GrafeoDB()
    assert db.import_csv(path, label="Raw", headers=False) == 2
    assert nodes(db, "Raw") == [
        {"column0": "Alix", "column1": "Amsterdam"},
        {"column0": "Gus", "column1": "Berlin", "column2": "3"},
    ]


def test_jsonl_keys_of_later_lines_keep_their_types(tmp_path):
    path = write(tmp_path, "events.jsonl", '{"id": 1}\n{"id": 2, "geo.lat": 52.5, "tags": ["x"]}\n')
    db = grafeo.GrafeoDB()
    assert db.import_jsonl(path, label="Event") == 2
    assert sorted(nodes(db, "Event"), key=lambda p: p["id"]) == [
        {"id": 1},
        {"id": 2, "geo.lat": 52.5, "tags": ["x"]},
    ]


def test_the_label_is_used_as_given(tmp_path):
    path = write(tmp_path, "fans.csv", "name\nMia\n")
    db = grafeo.GrafeoDB()
    assert db.import_csv(path, label="Film Fan") == 1
    assert nodes(db, "Film Fan") == [{"name": "Mia"}]


def test_the_count_is_what_the_import_created(tmp_path):
    path = write(tmp_path, "people.csv", "name\nVincent\nJules\n")
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Butch'})")
    assert db.import_csv(path, label="Person") == 2
    assert len(nodes(db, "Person")) == 3


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    db = grafeo.GrafeoDB()
    with pytest.raises(FileNotFoundError):
        db.import_csv(str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError):
        db.import_jsonl(str(tmp_path / "nope.jsonl"))


def test_an_empty_label_is_refused(tmp_path):
    path = write(tmp_path, "people.csv", "name\nHans\n")
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError):
        db.import_csv(path, label="")
