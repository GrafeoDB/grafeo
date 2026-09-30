"""shuffle_unordered: results without ORDER BY come back in random order."""

from grafeo import GrafeoDB


def values(db, query):
    return tuple(row["v"] for row in db.execute(query))


def test_shuffle_unordered_changes_unordered_results_only():
    db = GrafeoDB(shuffle_unordered=True)
    for v in range(50):
        db.execute(f"INSERT (:A {{v: {v}}})")
    orders = {values(db, "MATCH (n:A) RETURN n.v AS v") for _ in range(5)}
    assert len(orders) > 1
    assert all(sorted(order) == list(range(50)) for order in orders)
    ordered = {values(db, "MATCH (n:A) RETURN n.v AS v ORDER BY v") for _ in range(5)}
    assert ordered == {tuple(range(50))}


def test_shuffle_unordered_is_off_by_default():
    db = GrafeoDB()
    for v in range(50):
        db.execute(f"INSERT (:A {{v: {v}}})")
    assert len({values(db, "MATCH (n:A) RETURN n.v AS v") for _ in range(5)}) == 1
