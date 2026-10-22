"""GQL implementation of mutation tests.

Tests CRUD operations using GQL (ISO standard) query language.
"""

from tests.bases.test_mutations import BaseMutationsTest


class TestGQLMutations(BaseMutationsTest):
    """GQL implementation of mutation tests."""

    def create_node_query(self, labels: list[str], props: dict) -> str:
        """GQL: INSERT (:<labels> {<props>})"""
        label_str = ":".join(labels) if labels else ""
        if label_str:
            label_str = f":{label_str}"

        prop_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                prop_parts.append(f"{k}: '{v}'")
            elif isinstance(v, bool):
                prop_parts.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                prop_parts.append(f"{k}: null")
            else:
                prop_parts.append(f"{k}: {v}")

        prop_str = ", ".join(prop_parts)
        return f"INSERT (n{label_str} {{{prop_str}}}) RETURN n"

    def match_node_query(self, label: str, return_prop: str = "name") -> str:
        """GQL: MATCH (n:<label>) RETURN n.<prop>"""
        return f"MATCH (n:{label}) RETURN n.{return_prop}"

    def match_where_query(
        self, label: str, prop: str, op: str, value, return_prop: str = "name"
    ) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> <op> <value> RETURN n.<return_prop>"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {value_str} RETURN n.{return_prop}"

    def delete_node_query(self, label: str, prop: str, value) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> = <value> DELETE n"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} = {value_str} DELETE n"

    def create_edge_query(
        self,
        from_label: str,
        from_prop: str,
        from_value,
        to_label: str,
        to_prop: str,
        to_value,
        edge_type: str,
        edge_props: dict,
    ) -> str:
        """GQL: MATCH (a:<from_label>), (b:<to_label>) WHERE ... CREATE (a)-[:<edge_type>]->(b)"""
        from_val = f"'{from_value}'" if isinstance(from_value, str) else from_value
        to_val = f"'{to_value}'" if isinstance(to_value, str) else to_value

        prop_parts = []
        for k, v in edge_props.items():
            if isinstance(v, str):
                prop_parts.append(f"{k}: '{v}'")
            else:
                prop_parts.append(f"{k}: {v}")
        prop_str = ", ".join(prop_parts) if prop_parts else ""

        if prop_str:
            return (
                f"MATCH (a:{from_label}), (b:{to_label}) "
                f"WHERE a.{from_prop} = {from_val} AND b.{to_prop} = {to_val} "
                f"CREATE (a)-[r:{edge_type} {{{prop_str}}}]->(b) RETURN r"
            )
        else:
            return (
                f"MATCH (a:{from_label}), (b:{to_label}) "
                f"WHERE a.{from_prop} = {from_val} AND b.{to_prop} = {to_val} "
                f"CREATE (a)-[r:{edge_type}]->(b) RETURN r"
            )

    def update_node_query(
        self, label: str, match_prop: str, match_value, set_prop: str, set_value
    ) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> = <value> SET n.<prop> = <value>"""
        match_val = f"'{match_value}'" if isinstance(match_value, str) else match_value
        set_val = f"'{set_value}'" if isinstance(set_value, str) else set_value
        return (
            f"MATCH (n:{label}) WHERE n.{match_prop} = {match_val} "
            f"SET n.{set_prop} = {set_val} RETURN n"
        )


# =============================================================================
# GQL-SPECIFIC MUTATION TESTS
# =============================================================================


class TestGQLSpecificMutations:
    """GQL-specific mutation tests."""

    def test_gql_insert_syntax(self, db):
        """Test GQL INSERT syntax specifically."""
        result = db.execute("INSERT (:Person {name: 'InsertTest', age: 42}) RETURN *")
        rows = list(result)

        result = db.execute("MATCH (n:Person) WHERE n.name = 'InsertTest' RETURN n.age")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.age"] == 42

    def test_gql_multiple_labels_syntax(self, db):
        """Test GQL multiple labels syntax."""
        result = db.execute("INSERT (:Person:Developer:Senior {name: 'MultiLabel'}) RETURN *")
        list(result)

        result = db.execute("MATCH (n:Person:Developer) RETURN n.name")
        rows = list(result)
        assert len(rows) >= 1

    def test_gql_property_types(self, db):
        """Test various property types in GQL."""
        db.execute("INSERT (:Data {str: 'hello', num: 42, flt: 3.14, bool: true})")

        result = db.execute("MATCH (n:Data) RETURN n.str, n.num, n.flt, n.bool")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.str"] == "hello"
        assert rows[0]["n.num"] == 42

    def test_gql_set_multiple_properties(self, db):
        """Test SET with multiple properties."""
        db.execute("INSERT (:Person {name: 'Alix', age: 30, city: 'NYC'})")

        db.execute("MATCH (n:Person) WHERE n.name = 'Alix' SET n.age = 31, n.city = 'LA' RETURN n")

        result = db.execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.age, n.city")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.age"] == 31
        assert rows[0]["n.city"] == "LA"

    def test_gql_remove_property(self, db):
        """Test removing property by setting to null (REMOVE not yet in GQL)."""
        db.execute("INSERT (:Person {name: 'Gus', age: 25, temp: 'delete_me'})")

        # GQL uses SET n.prop = null to remove properties (REMOVE not yet implemented)
        db.execute("MATCH (n:Person) WHERE n.name = 'Gus' SET n.temp = null RETURN n")

        result = db.execute("MATCH (n:Person) WHERE n.name = 'Gus' RETURN n.temp")
        rows = list(result)
        # Property should be null or not present
        if len(rows) > 0:
            assert rows[0].get("n.temp") is None

    def test_gql_detach_delete(self, db):
        """Test DETACH DELETE (delete node and all relationships)."""
        alix = db.create_node(["Person"], {"name": "Alix"})
        gus = db.create_node(["Person"], {"name": "Gus"})
        db.create_edge(alix.id, gus.id, "KNOWS", {})

        # Delete Alix and her relationships
        db.execute("MATCH (n:Person) WHERE n.name = 'Alix' DETACH DELETE n")

        # Verify Alix is deleted
        result = db.execute("MATCH (n:Person) RETURN n.name")
        rows = list(result)
        names = [r["n.name"] for r in rows]
        assert "Alix" not in names
        assert "Gus" in names

    def test_gql_merge_create(self, db):
        """Test MERGE creates node if not exists."""
        db.execute("MERGE (:Person {name: 'MergeTest'})")

        result = db.execute("MATCH (n:Person) WHERE n.name = 'MergeTest' RETURN n.name")
        rows = list(result)
        assert len(rows) == 1

    def test_gql_merge_match(self, db):
        """Test MERGE matches existing node."""
        db.execute("INSERT (:Person {name: 'MergeExisting', age: 30})")

        # MERGE should match existing node, not create new
        db.execute("MERGE (n:Person {name: 'MergeExisting'}) SET n.age = 31 RETURN n")

        result = db.execute(
            "MATCH (n:Person) WHERE n.name = 'MergeExisting' RETURN count(n) AS cnt"
        )
        rows = list(result)
        assert rows[0]["cnt"] == 1

        result = db.execute("MATCH (n:Person) WHERE n.name = 'MergeExisting' RETURN n.age")
        rows = list(result)
        assert rows[0]["n.age"] == 31

    def test_gql_insert_negative_float(self, db):
        """Test INSERT with negative float property (issue #160)."""
        db.execute("INSERT (:Point {lat: -6.248, lon: 106.845})")

        result = db.execute("MATCH (n:Point) RETURN n.lat, n.lon")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.lat"] == -6.248
        assert rows[0]["n.lon"] == 106.845

    def test_gql_insert_negative_integer(self, db):
        """Test INSERT with negative integer property (issue #160)."""
        db.execute("INSERT (:Floor {level: -3})")

        result = db.execute("MATCH (n:Floor) RETURN n.level")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.level"] == -3

    def test_gql_insert_multiple_negative_properties(self, db):
        """Test INSERT with multiple negative properties (issue #160)."""
        db.execute("INSERT (:Point {lat: -6.248, lon: -106.845, alt: -42})")

        result = db.execute("MATCH (n:Point) RETURN n.lat, n.lon, n.alt")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.lat"] == -6.248
        assert rows[0]["n.lon"] == -106.845
        assert rows[0]["n.alt"] == -42

    def test_gql_insert_geographic_coordinates(self, db):
        """Test realistic lat/lon storage, the original issue #160 use case."""
        db.execute("INSERT (:City {name: 'Jakarta', lat: -6.175, lon: 106.827})")
        db.execute("INSERT (:City {name: 'Buenos Aires', lat: -34.604, lon: -58.382})")

        result = db.execute("MATCH (n:City {name: 'Buenos Aires'}) RETURN n.lat, n.lon")
        rows = list(result)
        assert rows[0]["n.lat"] == -34.604
        assert rows[0]["n.lon"] == -58.382

    def test_gql_negative_edge_property_inline(self, db):
        """Test negative property on an inline edge (issue #160)."""
        db.execute("INSERT (:Node {name: 'Alix'})-[:LINK {weight: -0.5}]->(:Node {name: 'Gus'})")

        result = db.execute("MATCH ()-[r:LINK]->() RETURN r.weight")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["r.weight"] == -0.5

    def test_gql_negative_edge_property_via_match(self, db):
        """Test negative property on edge created via MATCH/CREATE (issue #160)."""
        db.execute("INSERT (:Node {name: 'Alix'})")
        db.execute("INSERT (:Node {name: 'Gus'})")
        db.execute(
            "MATCH (a:Node {name: 'Alix'}), (b:Node {name: 'Gus'}) "
            "CREATE (a)-[:SCORED {score: -10}]->(b)"
        )

        result = db.execute("MATCH ()-[r:SCORED]->() RETURN r.score")
        rows = list(result)
        assert rows[0]["r.score"] == -10

    def test_gql_set_negative_integer(self, db):
        """Test SET with negative integer (issue #160)."""
        db.execute("INSERT (:Floor {name: 'Parking', level: 1})")
        db.execute("MATCH (n:Floor {name: 'Parking'}) SET n.level = -2")

        result = db.execute("MATCH (n:Floor {name: 'Parking'}) RETURN n.level")
        rows = list(result)
        assert rows[0]["n.level"] == -2

    def test_gql_set_negative_float(self, db):
        """Test SET with negative float (issue #160)."""
        db.execute("INSERT (:Sensor {name: 'Temp', reading: 25.0})")
        db.execute("MATCH (n:Sensor {name: 'Temp'}) SET n.reading = -18.5")

        result = db.execute("MATCH (n:Sensor {name: 'Temp'}) RETURN n.reading")
        rows = list(result)
        assert rows[0]["n.reading"] == -18.5

    def test_gql_where_equals_negative(self, db):
        """Test WHERE with negative equality (issue #160)."""
        db.execute("INSERT (:Floor {name: 'B1', level: -1})")
        db.execute("INSERT (:Floor {name: 'B2', level: -2})")
        db.execute("INSERT (:Floor {name: 'G', level: 0})")

        result = db.execute("MATCH (n:Floor) WHERE n.level = -2 RETURN n.name")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.name"] == "B2"

    def test_gql_where_less_than_negative(self, db):
        """Test WHERE with less-than negative comparison (issue #160)."""
        db.execute("INSERT (:Floor {name: 'B1', level: -1})")
        db.execute("INSERT (:Floor {name: 'B2', level: -2})")
        db.execute("INSERT (:Floor {name: 'B3', level: -3})")

        result = db.execute("MATCH (n:Floor) WHERE n.level < -1 RETURN n.name ORDER BY n.level")
        rows = list(result)
        assert len(rows) == 2

    def test_gql_where_between_negatives(self, db):
        """Test WHERE filtering between two negative values (issue #160)."""
        for val in [-10, -5, -1, 0, 5]:
            db.execute(f"INSERT (:Data {{val: {val}}})")

        result = db.execute("MATCH (n:Data) WHERE n.val >= -5 AND n.val <= -1 RETURN n.val")
        rows = list(result)
        assert len(rows) == 2

    def test_gql_property_map_match_negative(self, db):
        """Test inline property map matching with negative value (issue #160)."""
        db.execute("INSERT (:Floor {level: -1, name: 'Basement'})")
        db.execute("INSERT (:Floor {level: 1, name: 'Ground'})")

        result = db.execute("MATCH (n:Floor {level: -1}) RETURN n.name")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.name"] == "Basement"

    def test_gql_merge_create_with_negative(self, db):
        """Test MERGE creating node with negative property (issue #160)."""
        db.execute("MERGE (:Floor {level: -1, name: 'Basement'})")

        result = db.execute("MATCH (n:Floor) RETURN n.level, n.name")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.level"] == -1
        assert rows[0]["n.name"] == "Basement"

    def test_gql_merge_match_with_negative(self, db):
        """Test MERGE matching existing node with negative property (issue #160)."""
        db.execute("INSERT (:Floor {level: -1, name: 'Basement'})")
        db.execute("MERGE (n:Floor {level: -1}) ON MATCH SET n.name = 'SubBasement'")

        result = db.execute("MATCH (n:Floor) RETURN n.level, n.name")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["n.level"] == -1
        assert rows[0]["n.name"] == "SubBasement"

    def test_gql_merge_on_create_set_negative(self, db):
        """Test MERGE ON CREATE SET with negative value (issue #160)."""
        db.execute("MERGE (n:Sensor {name: 'Temp'}) ON CREATE SET n.reading = -40")

        result = db.execute("MATCH (n:Sensor {name: 'Temp'}) RETURN n.reading")
        rows = list(result)
        assert rows[0]["n.reading"] == -40

    def test_gql_merge_on_match_set_negative(self, db):
        """Test MERGE ON MATCH SET with negative value (issue #160)."""
        db.execute("INSERT (:Sensor {name: 'Temp', reading: 25})")
        db.execute("MERGE (n:Sensor {name: 'Temp'}) ON MATCH SET n.reading = -18")

        result = db.execute("MATCH (n:Sensor {name: 'Temp'}) RETURN n.reading")
        rows = list(result)
        assert rows[0]["n.reading"] == -18

    def test_gql_return_negative_literal(self, db):
        """Test RETURN with negative literal (issue #160)."""
        result = db.execute("RETURN -5 AS val")
        rows = list(result)
        assert rows[0]["val"] == -5

    def test_gql_negative_in_arithmetic(self, db):
        """Test negative value in arithmetic expression (issue #160)."""
        db.execute("INSERT (:Data {val: 10})")

        result = db.execute("MATCH (n:Data) RETURN n.val + -5 AS result")
        rows = list(result)
        assert rows[0]["result"] == 5
