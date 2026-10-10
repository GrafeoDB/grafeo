"""A damaged database file raises `GrafeoCorruptionError`.

A file Grafeo wrote whose bytes do not read back (a header, directory or chunk
checksum, a section that does not decode) raises `GrafeoCorruptionError`, a
`GrafeoError` with code `GRAFEO-S002` whose message names the file and, when
known, the byte. Opening it changes nothing on disk.
"""

import grafeo
import pytest


def write_database(path):
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Shosanna', city: 'Paris'})")
    db.close()


def flip(path, offset):
    with open(path, "r+b") as file:
        file.seek(offset)
        byte = file.read(1)[0]
        file.seek(offset)
        file.write(bytes([byte ^ 0x5A]))


def test_a_damaged_file_header_raises_grafeo_corruption_error(tmp_path):
    path = str(tmp_path / "paris.grafeo")
    write_database(path)
    # Inside the database id, which the header checksum covers.
    flip(path, 20)
    with open(path, "rb") as file:
        before = file.read()

    with pytest.raises(grafeo.GrafeoCorruptionError) as raised:
        grafeo.GrafeoDB(path)
    error = raised.value
    assert isinstance(error, grafeo.GrafeoError)
    assert error.error_code == "GRAFEO-S002"
    assert error.is_retryable is False
    message = str(error)
    assert "paris.grafeo" in message and "at byte 0" in message, message

    with open(path, "rb") as file:
        assert file.read() == before, "a refused open writes nothing"


def test_a_damaged_chunk_raises_grafeo_corruption_error_naming_its_byte(tmp_path):
    path = str(tmp_path / "prague.grafeo")
    write_database(path)
    with open(path, "rb") as file:
        at = file.read().index(b"Shosanna")
    flip(path, at)

    with pytest.raises(grafeo.GrafeoCorruptionError) as raised:
        grafeo.GrafeoDB(path)
    message = str(raised.value)
    assert f"at byte {at - at % 4096}" in message and "checksum" in message, message
