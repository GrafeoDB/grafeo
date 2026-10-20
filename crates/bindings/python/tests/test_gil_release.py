"""Long-running calls release the GIL, so other Python threads make progress.

Each test runs a call on a worker thread while the main thread spins a
counter, and compares what it counted during the call with what it counts on
its own in the same time. A call that keeps the GIL stops the counter for its
whole run, but for a switch interval or two around it (milliseconds, against
a call of at least ``MIN_SECONDS``); a call that releases it lets the counter
run on at its own pace. ``SHARE`` lies in between.
"""

import threading
import time

import pytest
from grafeo import GrafeoDB

# The worker's call runs at least this long.
MIN_SECONDS = 0.5
# The share of its own pace the main thread keeps while a call that releases
# the GIL runs, at least. One that keeps it leaves a few percent.
SHARE = 0.2


def long_count(db, size):
    """A query that runs for a while in Rust and returns ``size**2``."""
    return db.execute(
        f"UNWIND range(1, {size}) AS a UNWIND range(1, {size}) AS b RETURN count(*) AS c"
    )


def calibrated(call, prepare=None):
    """The smallest size (doubling) for which ``call(size)`` runs at least
    ``MIN_SECONDS``, after ``prepare(size)`` (not timed)."""
    size = 64
    for _ in range(16):
        if prepare is not None:
            prepare(size)
        start = time.perf_counter()
        call(size)
        if time.perf_counter() - start >= MIN_SECONDS:
            return size
        size *= 2
    pytest.fail(f"call never took {MIN_SECONDS}s (last size {size})")


def spin(counter, done):
    """Counts until ``done`` is set."""
    while not done.is_set():
        counter[0] += 1


def solo_rate():
    """Iterations per second the main thread counts on its own."""
    counter = [0]
    done = threading.Event()
    timer = threading.Timer(MIN_SECONDS / 2, done.set)
    start = time.perf_counter()
    timer.start()
    spin(counter, done)
    return counter[0] / (time.perf_counter() - start)


def share_during(call):
    """Runs ``call`` on a worker thread; returns its result and the share of
    its own pace the main thread kept while it ran."""
    rate = solo_rate()
    counter = [0]
    outcome = {}
    done = threading.Event()

    def work():
        try:
            before = counter[0]
            start = time.perf_counter()
            outcome["result"] = call()
            seconds = time.perf_counter() - start
            outcome["share"] = (counter[0] - before) / (rate * seconds)
        except BaseException as exc:  # noqa: BLE001 - reported below
            outcome["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    spin(counter, done)
    worker.join(timeout=60.0)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"], outcome["share"]


def concurrently(*calls):
    """Runs each call on its own thread; returns their results in order."""
    results = [None] * len(calls)
    errors = []

    def work(index, call):
        try:
            results[index] = call()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=work, args=(index, call), daemon=True)
        for index, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120.0)
    assert not errors, f"a concurrent call failed: {errors}"
    assert all(not thread.is_alive() for thread in threads), "a call did not finish (deadlock?)"
    return results


def count_of(result):
    return list(result)[0]["c"]


def test_query_releases_gil():
    db = GrafeoDB()
    size = calibrated(lambda size: long_count(db, size))
    result, share = share_during(lambda: long_count(db, size))
    assert count_of(result) == size * size
    assert share >= SHARE, f"main thread kept {share:.0%} of its pace while the query ran"


def test_concurrent_queries_are_both_correct():
    db = GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
    size = calibrated(lambda size: long_count(db, size))
    first, second = concurrently(
        lambda: long_count(db, size),
        lambda: db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name"),
    )
    assert count_of(first) == size * size
    assert [row["name"] for row in second] == ["Alix", "Gus"]


def test_transaction_execute_releases_gil():
    db = GrafeoDB()
    size = calibrated(lambda size: long_count(db, size))
    with db.begin_transaction() as tx:
        result, share = share_during(lambda: long_count(tx, size))
        assert count_of(result) == size * size
        assert share >= SHARE, f"main thread kept {share:.0%} of its pace while the query ran"


def test_concurrent_transactions_both_commit():
    db = GrafeoDB()
    size = calibrated(lambda size: long_count(db, size))

    def insert_and_count(name):
        tx = db.begin_transaction()
        tx.execute("INSERT (:Person {name: $name})", {"name": name})
        counted = count_of(long_count(tx, size))
        tx.commit()
        return counted

    assert concurrently(lambda: insert_and_count("Alix"), lambda: insert_and_count("Gus")) == [
        size * size,
        size * size,
    ]
    names = db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
    assert [row["name"] for row in names] == ["Alix", "Gus"]


def test_transaction_is_used_from_one_thread_at_a_time():
    """Committing while another thread runs the transaction's query raises at
    once (the query holds the transaction); the commit works once the query
    is done."""
    db = GrafeoDB()
    size = calibrated(lambda size: long_count(db, size))
    tx = db.begin_transaction()
    tx.execute("INSERT (:Person {name: 'Alix'})")
    started = threading.Event()

    def run():
        started.set()
        return count_of(long_count(tx, 2 * size))

    outcome = {}

    def work():
        try:
            outcome["count"] = run()
        except BaseException as exc:  # noqa: BLE001 - reported below
            outcome["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    started.wait()
    time.sleep(MIN_SECONDS / 4)
    with pytest.raises(RuntimeError, match="borrowed"):
        tx.commit()
    worker.join(timeout=60.0)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["count"] == 4 * size * size
    tx.commit()
    assert count_of(db.execute("MATCH (p:Person) RETURN count(p) AS c")) == 1


def ring(size):
    """A ring of ``size`` nodes with a chord from each node to the one
    halfway round."""
    db = GrafeoDB()
    ids = db.batch_create_nodes_with_props("Node", [{"index": i} for i in range(size)])
    edges = [(ids[i], ids[(i + 1) % size], "NEXT") for i in range(size)]
    edges += [(ids[i], ids[(i + size // 2) % size], "CHORD") for i in range(size)]
    db.batch_create_edges(edges)
    return db


@pytest.mark.skipif(
    not hasattr(GrafeoDB(), "algorithms"), reason="grafeo built without algos feature"
)
def test_algorithm_releases_gil():
    graphs = {}

    def prepare(size):
        if size not in graphs:
            graphs[size] = ring(size)

    def betweenness(size):
        return graphs[size].algorithms.betweenness_centrality()

    size = calibrated(betweenness, prepare)
    result, share = share_during(lambda: betweenness(size))
    assert len(result) == size
    assert share >= SHARE, f"main thread kept {share:.0%} of its pace while the algorithm ran"
    # Two runs at once give the same scores as one.
    first, second = concurrently(lambda: betweenness(size), lambda: betweenness(size))
    assert first == result
    assert second == result
