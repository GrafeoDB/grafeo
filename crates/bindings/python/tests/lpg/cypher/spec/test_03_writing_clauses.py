"""Cypher spec: Writing Clauses (openCypher 9 sec 3).

Covers: CREATE, DELETE, DETACH DELETE, SET, REMOVE, MERGE, FOREACH.
"""

# =============================================================================
# CREATE (sec 3.1)
# =============================================================================


class TestCreate:
    """CREATE clause variants."""

    def test_create_node(self, db):
        result = list(db.execute_cypher("CREATE (n:Person {name: 'Alix'}) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"

    def test_create_node_multi_label(self, db):
        result = list(db.execute_cypher("CREATE (n:Person:Developer {name: 'Alix'}) RETURN n.name"))
        assert result[0]["n.name"] == "Alix"
        # Verify both labels
        check = list(db.execute_cypher("MATCH (n:Person:Developer) RETURN n.name"))
        assert len(check) >= 1

    def test_create_relationship(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(
            db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
                "CREATE (a)-[r:KNOWS]->(b) RETURN type(r) AS t"
            )
        )
        assert result[0]["t"] == "KNOWS"

    def test_create_relationship_with_properties(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        result = list(
            db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
                "CREATE (a)-[r:KNOWS {since: 2020}]->(b) RETURN r.since"
            )
        )
        assert result[0]["r.since"] == 2020

    def test_create_path_pattern(self, db):
        """CREATE multi-hop path in one statement."""
        result = list(
            db.execute_cypher(
                "CREATE (a:Person {name: 'Alix'})-[:KNOWS]->"
                "(b:Person {name: 'Gus'})-[:KNOWS]->"
                "(c:Person {name: 'Vincent'}) "
                "RETURN a.name, b.name, c.name"
            )
        )
        assert len(result) == 1
        assert result[0]["c.name"] == "Vincent"


# =============================================================================
# DELETE (sec 3.2)
# =============================================================================


class TestDelete:
    """DELETE and DETACH DELETE."""

    def test_delete_node(self, db):
        db.create_node(["Temp"], {"v": 1})
        db.execute_cypher("MATCH (n:Temp) DELETE n")
        result = list(db.execute_cypher("MATCH (n:Temp) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 0

    def test_delete_multiple(self, db):
        db.create_node(["Temp"], {"v": 1})
        db.create_node(["Temp"], {"v": 2})
        db.execute_cypher("MATCH (n:Temp) DELETE n")
        result = list(db.execute_cypher("MATCH (n:Temp) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 0

    def test_detach_delete(self, db):
        """DETACH DELETE removes node and all connected edges."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
        result = list(db.execute_cypher("MATCH (n:Person) RETURN n.name"))
        names = {r["n.name"] for r in result}
        assert "Alix" not in names
        assert "Gus" in names

    def test_detach_delete_with_return(self, db):
        """DETACH DELETE n RETURN count(n) should return delete count (Deriva FR-3)."""
        a = db.create_node(["Person"], {"name": "Alix"})
        b = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(a.id, b.id, "KNOWS")
        result = list(
            db.execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) DETACH DELETE n RETURN count(n) AS deleted"
            )
        )
        assert result[0]["deleted"] == 1


# =============================================================================
# SET (sec 3.3)
# =============================================================================


class TestSet:
    """SET clause variants."""

    def test_set_property(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n.age = 30")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert result[0]["n.age"] == 30

    def test_set_multiple_properties(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n.age = 30, n.city = 'Amsterdam'")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age, n.city"))
        assert result[0]["n.age"] == 30
        assert result[0]["n.city"] == "Amsterdam"

    def test_set_replace_all(self, db):
        """SET n = {map} replaces all properties."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.execute_cypher(
            "MATCH (n:Person {name: 'Alix'}) SET n = {name: 'Alix', city: 'Amsterdam'}"
        )
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.city, n.age"))
        assert result[0]["n.city"] == "Amsterdam"
        assert result[0]["n.age"] is None  # age was replaced away

    def test_set_merge_map(self, db):
        """SET n += {map} merges properties."""
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n += {city: 'Amsterdam'}")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age, n.city"))
        assert result[0]["n.age"] == 30  # preserved
        assert result[0]["n.city"] == "Amsterdam"

    def test_set_label(self, db):
        """SET n:Label adds a label."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n:Developer")
        result = list(db.execute_cypher("MATCH (n:Developer) RETURN n.name"))
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_set_multiple_labels(self, db):
        """SET n:L1:L2 adds multiple labels."""
        db.create_node(["Person"], {"name": "Alix"})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n:Developer:Senior")
        result = list(db.execute_cypher("MATCH (n:Developer:Senior) RETURN n.name"))
        assert len(result) >= 1

    def test_set_label_preserves_variable_binding(self, db):
        """SET n:Label must not drop variable for subsequent clauses (#178)."""
        db.create_node(["Person"], {"name": "Alix"})
        result = list(
            db.execute_cypher("MATCH (n:Person {name: 'Alix'}) SET n:Employee RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"

    def test_set_label_then_remove_property(self, db):
        """SET n:Label then REMOVE n.prop on same variable (#178)."""
        db.create_node(["AWSAccount"], {"id": "123", "foreign": True})
        db.execute_cypher("MATCH (n:AWSAccount {id: '123'}) SET n:Tenant REMOVE n.foreign")
        result = list(db.execute_cypher("MATCH (n:Tenant) RETURN n.id"))
        assert len(result) == 1
        assert result[0]["n.id"] == "123"

    def test_count_star_after_set_label(self, db):
        """count(*) after SET n:Label returns correct count (#182)."""
        db.create_node(["Node"], {"id": "1"})
        db.create_node(["Node"], {"id": "2"})
        db.create_node(["Node"], {"id": "3"})
        result = list(db.execute_cypher("MATCH (n:Node) SET n:Tagged RETURN count(*) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_count_var_after_set_label(self, db):
        """count(n) after SET n:Label returns correct count (#182)."""
        db.create_node(["Node"], {"id": "1"})
        db.create_node(["Node"], {"id": "2"})
        result = list(db.execute_cypher("MATCH (n:Node) SET n:Tagged RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 2

    def test_set_property_to_timestamp(self, db):
        """SET n.prop = timestamp() stores non-null millis (#179)."""
        db.create_node(["Event"], {"name": "launch"})
        db.execute_cypher("MATCH (e:Event) SET e.created_at = timestamp()")
        result = list(db.execute_cypher("MATCH (e:Event) RETURN e.created_at"))
        assert result[0]["e.created_at"] is not None
        assert result[0]["e.created_at"] > 1_577_836_800_000


# =============================================================================
# REMOVE (sec 3.4)
# =============================================================================


class TestRemove:
    """REMOVE clause variants."""

    def test_remove_property(self, db):
        db.create_node(["Person"], {"name": "Alix", "age": 30})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) REMOVE n.age")
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.age"))
        assert result[0]["n.age"] is None

    def test_remove_label(self, db):
        db.create_node(["Person", "Developer"], {"name": "Alix"})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) REMOVE n:Developer")
        result = list(db.execute_cypher("MATCH (n:Developer) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 0

    def test_remove_multiple_labels(self, db):
        db.create_node(["Person", "Developer", "Senior"], {"name": "Alix"})
        db.execute_cypher("MATCH (n:Person {name: 'Alix'}) REMOVE n:Developer:Senior")
        result = list(db.execute_cypher("MATCH (n:Developer) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 0

    def test_remove_label_preserves_variable_binding(self, db):
        """REMOVE n:Label must not drop variable for subsequent clauses (#178)."""
        db.create_node(["Person", "Employee"], {"name": "Alix"})
        result = list(
            db.execute_cypher("MATCH (n:Person {name: 'Alix'}) REMOVE n:Employee RETURN n.name")
        )
        assert len(result) == 1
        assert result[0]["n.name"] == "Alix"


# =============================================================================
# MATCH + CREATE edge (phantom node regression, #181)
# =============================================================================


class TestCreateEdgePhantom:
    """Verify MATCH...CREATE edge does not create phantom nodes (#181)."""

    def test_match_create_edge_no_phantoms(self, db):
        """Creating edge between matched nodes must not create extra nodes."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        before = list(db.execute_cypher("MATCH (n) RETURN count(n) AS cnt"))
        count_before = before[0]["cnt"]

        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)"
        )

        after = list(db.execute_cypher("MATCH (n) RETURN count(n) AS cnt"))
        assert after[0]["cnt"] == count_before, "phantom nodes were created"

        edges = list(db.execute_cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt"))
        assert edges[0]["cnt"] == 1

    def test_match_create_multiple_edges_no_phantoms(self, db):
        """Multiple edges between matched nodes must not multiply phantoms."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.create_node(["Person"], {"name": "Mia"})

        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)"
        )
        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (c:Person {name: 'Mia'}) CREATE (a)-[:KNOWS]->(c)"
        )

        nodes = list(db.execute_cypher("MATCH (n) RETURN count(n) AS cnt"))
        assert nodes[0]["cnt"] == 3

        edges = list(db.execute_cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt"))
        assert edges[0]["cnt"] == 2

    def test_match_create_edge_correct_endpoints(self, db):
        """Edge connects the correct nodes."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)"
        )
        result = list(
            db.execute_cypher("MATCH (s)-[:KNOWS]->(t) RETURN s.name AS src, t.name AS dst")
        )
        assert len(result) == 1
        assert result[0]["src"] == "Alix"
        assert result[0]["dst"] == "Gus"

    def test_startnode_endnode(self, db):
        """startNode(r) and endNode(r) return the nodes, as in openCypher (0.6.0; IDs before)."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        result = list(
            db.execute_cypher("MATCH ()-[r:KNOWS]->() RETURN startNode(r) AS sn, endNode(r) AS en")
        )
        assert len(result) == 1
        start, end = result[0]["sn"], result[0]["en"]
        assert start["name"] == "Alix" and start["_labels"] == ["Person"]
        assert end["name"] == "Gus" and end["_labels"] == ["Person"]
        assert start["_id"] != end["_id"]

    def test_startnode_equals_source(self, db):
        """startNode(r) == id(s) and endNode(r) == id(t) (#180)."""
        db.execute_cypher("CREATE (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        result = list(
            db.execute_cypher(
                "MATCH (s)-[r:KNOWS]->(t) "
                "RETURN startNode(r) = id(s) AS src_ok, endNode(r) = id(t) AS dst_ok"
            )
        )
        assert result[0]["src_ok"] is True
        assert result[0]["dst_ok"] is True

    def test_timestamp_returns_millis(self, db):
        """timestamp() returns millisecond epoch as integer (#179)."""
        result = list(db.execute_cypher("RETURN timestamp() AS ts"))
        assert len(result) == 1
        assert result[0]["ts"] is not None
        assert isinstance(result[0]["ts"], int)
        assert result[0]["ts"] > 1_577_836_800_000


# =============================================================================
# MERGE (sec 3.5)
# =============================================================================


class TestMerge:
    """MERGE clause variants."""

    def test_merge_create(self, db):
        """MERGE creates when no match."""
        list(db.execute_cypher("MERGE (n:Person {name: 'Alix'})"))
        result = list(db.execute_cypher("MATCH (n:Person) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1

    def test_merge_match(self, db):
        """MERGE finds existing node."""
        db.create_node(["Person"], {"name": "Alix"})
        list(db.execute_cypher("MERGE (n:Person {name: 'Alix'})"))
        result = list(db.execute_cypher("MATCH (n:Person) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 1  # No duplicate

    def test_merge_on_create_set(self, db):
        """MERGE ON CREATE SET sets properties only when creating."""
        list(db.execute_cypher("MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = true"))
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.created"))
        assert result[0]["n.created"] is True

    def test_merge_on_match_set(self, db):
        """MERGE ON MATCH SET updates properties when matching."""
        db.create_node(["Person"], {"name": "Alix", "visits": 1})
        list(db.execute_cypher("MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.visits = 2"))
        result = list(db.execute_cypher("MATCH (n:Person {name: 'Alix'}) RETURN n.visits"))
        assert result[0]["n.visits"] == 2

    def test_merge_relationship(self, db):
        """MERGE on relationship."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        list(
            db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)-[:KNOWS]->(b)"
            )
        )
        result = list(
            db.execute_cypher("MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) RETURN b.name")
        )
        assert result[0]["b.name"] == "Gus"

    def test_merge_relationship_set(self, db):
        """MATCH + MERGE (a)-[r:REL]->(b) SET r.prop works with pre-matched nodes."""
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        list(
            db.execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
                "MERGE (a)-[r:KNOWS {since: 2020}]->(b) SET r.weight = 0.5"
            )
        )
        result = list(
            db.execute_cypher(
                "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) RETURN r.weight"
            )
        )
        assert result[0]["r.weight"] == 0.5

    def test_merge_inline_relationship_set(self, db):
        """MERGE (a:L)-[r:REL]->(b:L) SET r.prop with inline node creation (Deriva FR-4)."""
        list(
            db.execute_cypher(
                "MERGE (a:M {id: 1})-[r:REL {type: 'test'}]->(b:M {id: 2}) SET r.weight = 0.5"
            )
        )
        result = list(db.execute_cypher("MATCH (:M {id: 1})-[r:REL]->(:M {id: 2}) RETURN r.weight"))
        assert result[0]["r.weight"] == 0.5


# =============================================================================
# FOREACH (sec 3.6)
# =============================================================================


class TestForeach:
    """FOREACH clause."""

    def test_foreach_create(self, db):
        """FOREACH (x IN list | CREATE ...) with preceding MATCH."""
        db.create_node(["Anchor"], {"v": 1})
        list(
            db.execute_cypher(
                "MATCH (a:Anchor) "
                "FOREACH (name IN ['Alix', 'Gus', 'Vincent'] | "
                "  CREATE (:Person {name: name})"
                ")"
            )
        )
        result = list(db.execute_cypher("MATCH (n:Person) RETURN count(n) AS cnt"))
        assert result[0]["cnt"] == 3

    def test_foreach_set(self, db):
        """FOREACH with SET mutation."""
        db.create_node(["Person"], {"name": "Alix", "active": False})
        db.create_node(["Person"], {"name": "Gus", "active": False})
        list(
            db.execute_cypher(
                "MATCH (n:Person) "
                "WITH collect(n) AS people "
                "FOREACH (p IN people | SET p.active = true)"
            )
        )
        result = list(
            db.execute_cypher("MATCH (n:Person) WHERE n.active = true RETURN count(n) AS cnt")
        )
        assert result[0]["cnt"] == 2
