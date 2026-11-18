"""Deleted values leave the memory Grafeo allocated (#712), the reporter's scenario in Python.

Values holding a marker are written, deleted, compacted and the database closed; the
process's own memory then holds no copy of the marker. The check runs in a child process
where glibc overwrites every block it frees (``MALLOC_PERTURB_``) and Python allocates its
small objects with malloc too (``PYTHONMALLOC=malloc``), so freed memory holds no marker.

The marker exists whole only inside Grafeo: the query joins its two halves with ``||``, and
the scan looks for the first half followed by the second, so no Python object ever holds
it. The scan reads the process's read-write mappings (``/proc/self/maps``,
``/proc/self/mem``) into one buffer made before the negative control, and skips that buffer,
so what it reads never shows up in a later count.

Linux with glibc only: elsewhere freed memory keeps its bytes.
"""

import ctypes
import gc
import json
import mmap
import os
import platform
import subprocess
import sys
import time

import pytest
from grafeo import GrafeoDB

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or platform.libc_ver()[0] != "glibc",
    reason="needs glibc's MALLOC_PERTURB_, which overwrites the blocks it frees",
)

# The size of the scan's buffer: a mapping is read in pieces of this size.
CHUNK = 1 << 20


def scanner(first, second):
    """A function counting the marker (`first` then `second`) in this process's memory."""
    buffer = mmap.mmap(-1, CHUNK)
    own = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    overlap = len(first) + len(second) - 1

    def regions():
        with open("/proc/self/maps", encoding="ascii") as maps:
            for line in maps:
                fields = line.split()
                start, end = (int(part, 16) for part in fields[0].split("-"))
                if "rw" not in fields[1]:
                    continue
                if len(fields) > 5 and fields[5] in ("[vvar]", "[vsyscall]", "[vdso]"):
                    continue
                # Only the buffer's own bytes are skipped: a mapping merged with it is
                # scanned around it.
                for piece in ((start, min(end, own)), (max(start, own + CHUNK), end)):
                    if piece[0] < piece[1]:
                        yield piece

    def count():
        found = 0
        with open("/proc/self/mem", "rb", buffering=0) as memory:
            for start, end in list(regions()):
                at = start
                while at < end:
                    size = min(CHUNK, end - at)
                    try:
                        memory.seek(at)
                        read = memory.readinto(memoryview(buffer)[:size])
                    except OSError:
                        break
                    if not read:
                        break
                    position = buffer.find(first, 0, read)
                    while position != -1:
                        tail = position + len(first)
                        end_of_marker = tail + len(second)
                        if end_of_marker <= read and buffer[tail:end_of_marker] == second:
                            found += 1
                        position = buffer.find(first, position + 1, read)
                    if at + read >= end:
                        break
                    # Back by the overlap, so a marker across two pieces is found once.
                    at += max(read - overlap, 1)
        return found

    return count


def child(directory):
    """The scenario, run in the child: the counts after each step, as JSON."""
    first = f"zq{os.getpid()}x{time.time_ns()}".encode()
    second = b"vexmarrow"
    count = scanner(first, second)
    counts = {}

    db = GrafeoDB(os.path.join(directory, "deleted.grafeo"))
    for item in range(20):
        db.execute(
            "INSERT (:Item {id: $id, text: $text})",
            {"id": f"I{item}", "text": f"Item {item}: Gus said something."},
        )
    counts["before"] = count()
    for item in range(3):
        db.execute(
            "INSERT (:Item {id: $id, text: $first || $second})",
            {"id": f"M{item}", "first": first.decode(), "second": second.decode()},
        )
    counts["written"] = count()
    db.execute("MATCH (n:Item) WHERE n.id IN ['M0', 'M1', 'M2'] DETACH DELETE n")
    db.compact()
    db.close()
    counts["closed"] = count()
    del db
    gc.collect()
    counts["dropped"] = count()
    print(json.dumps(counts))


def test_deleted_values_leave_the_memory_grafeo_allocated(tmp_path):
    environment = dict(os.environ, MALLOC_PERTURB_="165", PYTHONMALLOC="malloc")
    run = subprocess.run(
        [sys.executable, __file__, str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    counts = json.loads(run.stdout.strip().splitlines()[-1])
    assert counts["before"] == 0, f"the negative control: no copy yet ({counts})"
    assert counts["written"] > 0, f"the positive control: the scan finds the values ({counts})"
    assert counts["closed"] == 0, f"after delete, compact() and close() ({counts})"
    assert counts["dropped"] == 0, f"after the database object is gone ({counts})"


if __name__ == "__main__":
    child(sys.argv[1])
