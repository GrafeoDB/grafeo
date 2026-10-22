"""Cypher spec: Administration and Schema (openCypher 9 sec 8, 9).

Covers: CREATE/DROP INDEX, CREATE/DROP CONSTRAINT, SHOW commands,
EXPLAIN, PROFILE. The xfail cases are not implemented yet.
"""

import pytest

# =============================================================================
# Index DDL (sec 8)
# =============================================================================


class TestIndex:
    """CREATE/DROP INDEX (not implemented in Cypher parser)."""

    @pytest.mark.xfail(reason="Index DDL not implemented in Cypher")
    def test_create_index(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.execute_cypher("CREATE INDEX FOR (n:Person) ON (n.name)")

    @pytest.mark.xfail(reason="Index DDL not implemented in Cypher")
    def test_create_index_if_not_exists(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.execute_cypher("CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.name)")

    @pytest.mark.xfail(reason="Index DDL not implemented in Cypher")
    def test_drop_index(self, db):
        db.execute_cypher("DROP INDEX my_index")

    def test_drop_index_if_exists(self, db):
        db.execute_cypher("DROP INDEX my_index IF EXISTS")


# =============================================================================
# Constraint DDL (sec 8)
# =============================================================================


class TestConstraint:
    """CREATE/DROP CONSTRAINT (Cypher requires a constraint name)."""

    @pytest.mark.xfail(reason="Cypher CREATE CONSTRAINT requires a name")
    def test_create_unique_constraint(self, db):
        db.execute_cypher("CREATE CONSTRAINT FOR (n:Person) REQUIRE n.email IS UNIQUE")

    @pytest.mark.xfail(reason="Cypher CREATE CONSTRAINT requires a name")
    def test_create_exists_constraint(self, db):
        db.execute_cypher("CREATE CONSTRAINT FOR (n:Person) REQUIRE n.name IS NOT NULL")

    def test_drop_constraint(self, db):
        db.execute_cypher("CREATE CONSTRAINT unique_email FOR (p:Person) REQUIRE p.email IS UNIQUE")
        db.execute_cypher("DROP CONSTRAINT unique_email")
        assert list(db.execute_cypher("SHOW CONSTRAINTS")) == []
        # Duplicates are accepted again once the constraint is gone.
        db.execute_cypher("CREATE (:Person {email: 'alix@example.com'})")
        db.execute_cypher("CREATE (:Person {email: 'alix@example.com'})")

    def test_drop_unknown_constraint_fails(self, db):
        with pytest.raises(Exception, match="constraint 'my_constraint' does not exist"):
            db.execute_cypher("DROP CONSTRAINT my_constraint")

    def test_drop_constraint_if_exists(self, db):
        db.execute_cypher("DROP CONSTRAINT my_constraint IF EXISTS")


# =============================================================================
# SHOW commands (sec 8)
# =============================================================================


class TestShowCommands:
    """SHOW INDEXES, SHOW CONSTRAINTS, etc. (not implemented)."""

    def test_show_indexes(self, db):
        result = list(db.execute_cypher("SHOW INDEXES"))
        assert isinstance(result, list)

    def test_show_constraints(self, db):
        result = list(db.execute_cypher("SHOW CONSTRAINTS"))
        assert isinstance(result, list)

    @pytest.mark.xfail(reason="SHOW commands not implemented")
    def test_show_procedures(self, db):
        result = list(db.execute_cypher("SHOW PROCEDURES"))
        assert isinstance(result, list)

    @pytest.mark.xfail(reason="SHOW commands not implemented")
    def test_show_functions(self, db):
        result = list(db.execute_cypher("SHOW FUNCTIONS"))
        assert isinstance(result, list)


# =============================================================================
# Query Analysis (sec 9)
# =============================================================================


class TestQueryAnalysis:
    """EXPLAIN and PROFILE (not implemented)."""

    def test_explain(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("EXPLAIN MATCH (n:Person) RETURN n.name"))
        assert result is not None

    def test_profile(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        result = list(db.execute_cypher("PROFILE MATCH (n:Person) RETURN n.name"))
        assert result is not None


# =============================================================================
# Syntax / Lexer (sec 10)
# =============================================================================


class TestSyntax:
    """Lexer features: comments, identifiers, escape sequences."""

    def test_line_comment(self, db):
        """// line comment."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) // this is a comment\nRETURN n.v"))
        assert result[0]["n.v"] == 1

    def test_block_comment(self, db):
        """/* block comment */."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) /* comment */ RETURN n.v"))
        assert result[0]["n.v"] == 1

    def test_backtick_identifier(self, db):
        """Backtick-quoted identifiers."""
        db.create_node(["N"], {"special key": "val"})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.`special key` AS r"))
        assert result[0]["r"] == "val"

    def test_semicolons(self, db):
        """Semicolons as statement separator."""
        db.create_node(["N"], {"v": 1})
        result = list(db.execute_cypher("MATCH (n:N) RETURN n.v;"))
        assert result[0]["n.v"] == 1
