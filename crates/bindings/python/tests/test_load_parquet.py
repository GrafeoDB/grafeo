"""`LOAD DATA ... FORMAT PARQUET` reads a Parquet file from Python.

The wheel's `full` feature left out `parquet-import`, so the statement the GQL
guide shows failed in every installed wheel. Uncompressed and Snappy files
load (Snappy is the default of pyarrow, pandas and Spark); the other codecs
are refused with an error that names the codec.
"""

import grafeo
import pytest


def test_the_wheel_imports_parquet():
    assert "parquet-import" in grafeo.features()


def write_people(tmp_path, **options):
    """Alix, Gus and Mia as a Parquet file, written with pyarrow `options`."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "name": ["Alix", "Gus", "Mia"],
            "city": ["Amsterdam", None, "Paris"],
            "age": pa.array([31, 42, 28], type=pa.int64()),
            "score": [1.5, 2.25, -0.5],
            "member": [True, False, True],
        }
    )
    path = tmp_path / "people.parquet"
    pq.write_table(table, path, **options)
    codecs = {
        pq.ParquetFile(path).metadata.row_group(0).column(index).compression
        for index in range(table.num_columns)
    }
    return path, codecs


def load_people(path):
    db = grafeo.GrafeoDB()
    # Forward slashes: a backslash in a GQL string starts an escape.
    db.execute(
        f"LOAD DATA FROM '{path.as_posix()}' FORMAT PARQUET AS row "
        "INSERT (:Person {name: row.name, city: row.city, age: row.age, "
        "score: row.score, member: row.member})"
    )
    return db


@pytest.mark.parametrize(
    ("options", "codec"),
    [({"compression": "none"}, "UNCOMPRESSED"), ({}, "SNAPPY")],
    ids=["uncompressed", "snappy-by-default"],
)
def test_load_data_reads_a_parquet_file(tmp_path, options, codec):
    path, codecs = write_people(tmp_path, **options)
    # pyarrow wrote what the case names: with no option, Snappy.
    assert codecs == {codec}
    rows = load_people(path).execute(
        "MATCH (p:Person) RETURN p.name AS name, p.city AS city, p.age AS age, "
        "p.score AS score, p.member AS member ORDER BY name"
    )
    assert [(r["name"], r["city"], r["age"], r["score"], r["member"]) for r in rows] == [
        ("Alix", "Amsterdam", 31, 1.5, True),
        ("Gus", None, 42, 2.25, False),
        ("Mia", "Paris", 28, -0.5, True),
    ]


@pytest.mark.parametrize("compression", ["zstd", "gzip"])
def test_another_codec_is_refused_naming_it(tmp_path, compression):
    path, codecs = write_people(tmp_path, compression=compression)
    assert codecs == {compression.upper()}
    # The build has Snappy only: gzip is flate2 in the message.
    named = {"zstd": "zstd", "gzip": "flate2"}[compression]
    with pytest.raises(grafeo.GrafeoError, match=named):
        load_people(path)
