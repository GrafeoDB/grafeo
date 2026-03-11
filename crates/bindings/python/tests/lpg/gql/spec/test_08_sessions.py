"""GQL spec: Sessions and Transactions (ISO sec 6, 7, 8).

Covers: START TRANSACTION, COMMIT, ROLLBACK, USE GRAPH,
SESSION SET (GRAPH, TIME ZONE, SCHEMA, PARAMETER), SESSION RESET,
SESSION CLOSE, transaction characteristics, savepoints, nested transactions.
"""


# =============================================================================
# Basic Transaction Control (sec 6)
# =============================================================================


class TestTransactions:
    """begin_transaction, commit, rollback."""

    def test_commit_persists(self, db):
        """Committed changes are visible."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Alix'})")
            tx.commit()
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        assert any(r["n.name"] == "Alix" for r in result)

    def test_rollback_discards(self, db):
        """Rolled-back changes are not visible."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Ghost'})")
            tx.rollback()
        result = list(db.execute("MATCH (n:Person {name: 'Ghost'}) RETURN n"))
        assert len(result) == 0

    def test_read_only_transaction(self, db):
        """Transaction used for read-only access."""
        db.create_node(["Person"], {"name": "Alix"})
        with db.begin_transaction() as tx:
            result = list(tx.execute("MATCH (n:Person) RETURN n.name"))
            assert len(result) == 1
            tx.commit()

    def test_read_write_transaction(self, db):
        """Transaction with write followed by commit."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.commit()
        result = list(db.execute("MATCH (n:Person {name: 'Gus'}) RETURN n.name"))
        assert len(result) == 1


# =============================================================================
# Savepoints (sec 6)
# =============================================================================


class TestSavepoints:
    """SAVEPOINT, ROLLBACK TO SAVEPOINT, RELEASE SAVEPOINT."""

    def test_savepoint_and_rollback_to(self, db):
        """SAVEPOINT + ROLLBACK TO undoes partial work."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Alix'})")
            tx.execute("SAVEPOINT sp1")
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.execute("ROLLBACK TO SAVEPOINT sp1")
            tx.commit()
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert "Alix" in names
        assert "Gus" not in names

    def test_release_savepoint(self, db):
        """RELEASE SAVEPOINT merges savepoint state."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Alix'})")
            tx.execute("SAVEPOINT sp1")
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.execute("RELEASE SAVEPOINT sp1")
            tx.commit()
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert "Alix" in names
        assert "Gus" in names


# =============================================================================
# Nested Transactions (sec 6)
# =============================================================================


class TestNestedTransactions:
    """Nested transaction behavior via context managers."""

    def test_nested_commit(self, db):
        """Inner COMMIT releases auto-savepoint."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Alix'})")
            tx.execute("START TRANSACTION")  # nested
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.execute("COMMIT")  # inner
            tx.commit()  # outer
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert "Alix" in names
        assert "Gus" in names

    def test_nested_rollback(self, db):
        """Inner ROLLBACK undoes only inner work."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Alix'})")
            tx.execute("START TRANSACTION")  # nested
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.execute("ROLLBACK")  # inner
            tx.commit()  # outer
        result = list(db.execute("MATCH (n:Person) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert "Alix" in names
        assert "Gus" not in names


# =============================================================================
# Session Management (sec 7, 8)
# =============================================================================


class TestSessionManagement:
    """SESSION SET, SESSION RESET, SESSION CLOSE."""

    def test_session_set_graph(self, db):
        """SESSION SET GRAPH name via db.execute."""
        db.execute("CREATE GRAPH session_graph")
        db.execute("SESSION SET GRAPH session_graph")

    def test_session_set_time_zone(self, db):
        """SESSION SET TIME ZONE via db.execute."""
        db.execute("SESSION SET TIME ZONE '+01:00'")

    def test_session_set_schema(self, db):
        """SESSION SET SCHEMA name via db.execute."""
        db.execute("CREATE GRAPH my_ns")
        db.execute("SESSION SET SCHEMA my_ns")

    def test_session_set_parameter(self, db):
        """SESSION SET PARAMETER $name = value via db.execute."""
        db.execute("SESSION SET PARAMETER $my_param = 42")

    def test_session_reset(self, db):
        """SESSION RESET restores defaults."""
        db.execute("SESSION SET TIME ZONE '+02:00'")
        db.execute("SESSION RESET")

    def test_session_reset_all(self, db):
        """SESSION RESET ALL restores all defaults."""
        db.execute("SESSION RESET ALL")

    def test_session_close(self, db):
        """SESSION CLOSE terminates the session."""
        db.execute("SESSION CLOSE")
