"""The difftest corpus: fixtures and the queries that run on them.

The fixtures are built so that mistakes show. Nodes carry a property `w` of 100 and
up, edges one below 100, so an edge read as a node (or the reverse) gives a value from
the wrong range. `chain` has 5,000 nodes, so sorts, cuts and skips cross row batches.

Each case has an id, the query, the languages it runs in, its fixture, and whether its
rows are ordered (unordered rows compare as a multiset). Queries only read: each fixture
is built once and shared by its cases. Results are matched by id and
language, so never change or reuse an id: add new cases at the end of a section, or
start a new section.
"""

from __future__ import annotations

from dataclasses import dataclass

GQL = ("gql",)
CYPHER = ("cypher",)
BOTH = ("gql", "cypher")
LANGUAGES = frozenset(BOTH)


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    languages: tuple[str, ...] = BOTH
    fixture: str = "social"
    ordered: bool = False


def social(grafeo):
    """Five people and three cities. KNOWS: a triangle Alix, Gus, Vincent, plus
    Alix to Jules and Jules to Mia. LIVES_IN: Alix in Amsterdam, Gus in Berlin, Mia in
    Paris. Jules, Mia and Berlin have no `w`."""
    db = grafeo.GrafeoDB()
    people = [
        ("Alix", 30, 100),
        ("Gus", 25, 101),
        ("Vincent", 40, 103),
        ("Jules", 35, None),
        ("Mia", 28, None),
    ]
    for name, age, w in people:
        props = f"name: '{name}', age: {age}" + (f", w: {w}" if w is not None else "")
        db.execute(f"INSERT (:Person {{{props}}})")
    for name, w in [("Amsterdam", 104), ("Berlin", None), ("Paris", 105)]:
        props = f"name: '{name}'" + (f", w: {w}" if w is not None else "")
        db.execute(f"INSERT (:City {{{props}}})")
    knows = [
        ("Alix", "Gus", 2010, 1),
        ("Gus", "Vincent", 2012, 2),
        ("Vincent", "Alix", 2015, 3),
        ("Jules", "Mia", 2020, 4),
        ("Alix", "Jules", 2018, 5),
    ]
    for a, b, since, w in knows:
        db.execute(
            f"MATCH (a:Person {{name: '{a}'}}), (b:Person {{name: '{b}'}}) "
            f"INSERT (a)-[:KNOWS {{since: {since}, w: {w}}}]->(b)"
        )
    lives = [
        ("Alix", "Amsterdam", 5, 6),
        ("Gus", "Berlin", 3, 7),
        ("Mia", "Paris", 1, 8),
    ]
    for a, c, years, w in lives:
        db.execute(
            f"MATCH (a:Person {{name: '{a}'}}), (c:City {{name: '{c}'}}) "
            f"INSERT (a)-[:LIVES_IN {{years: {years}, w: {w}}}]->(c)"
        )
    return db


def chain(grafeo):
    """5,000 `N` nodes (`i` 0 to 4999, `m` = i % 7) linked by NEXT edges (`k` = i)."""
    db = grafeo.GrafeoDB()
    db.execute("UNWIND range(0, 4999) AS i INSERT (:N {i: i, m: i % 7})")
    db.execute(
        "MATCH (a:N), (b:N) WHERE b.i = a.i + 1 INSERT (a)-[:NEXT {k: a.i}]->(b)"
    )
    return db


def empty(grafeo):
    return grafeo.GrafeoDB()


FIXTURES = {"social": social, "chain": chain, "empty": empty}

CASES: list[Case] = []


def case(
    case_id: str,
    query: str,
    languages: tuple[str, ...] = BOTH,
    fixture: str = "social",
    ordered: bool = False,
) -> None:
    CASES.append(Case(case_id, query, languages, fixture, ordered))


def ordered(
    case_id: str, query: str, languages: tuple[str, ...] = BOTH, fixture: str = "social"
):
    case(case_id, query, languages, fixture, ordered=True)


# The cases, one per line.
# fmt: off

# A: nodes and edges returned through ORDER BY, LIMIT, SKIP and DISTINCT
ordered("A1", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN r ORDER BY r.since")
ordered("A2", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN r ORDER BY r.since DESC LIMIT 2")
ordered("A3", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN r ORDER BY r.since SKIP 2")
ordered("A4", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN r ORDER BY r.since SKIP 1 LIMIT 2")
case("A5", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN r SKIP 1 LIMIT 2")
ordered("A6", "MATCH (a:Person)-[r]->(b) RETURN DISTINCT type(r) AS t ORDER BY t")
ordered("A7", "MATCH (a:Person)-[r]->(b) RETURN DISTINCT r ORDER BY r.w")
ordered("A8", "MATCH (a:Person)-[r]->(b) RETURN a, r, b ORDER BY r.w LIMIT 3")
ordered("A9", "MATCH p = (a:Person)-[:KNOWS]->(b) RETURN p ORDER BY a.name LIMIT 2")
ordered("A10", "MATCH (a:Person) RETURN a ORDER BY a.age DESC SKIP 1 LIMIT 2")
ordered("A11", "MATCH (a:Person)-[r]->(b) RETURN r ORDER BY type(r), r.w")
ordered("A12", "MATCH (a:Person)-[r]->(b) RETURN r, b ORDER BY b.name, r.w")
ordered("A13", "MATCH (a:Person)-[r*1..2]->(b) RETURN r ORDER BY size(r), b.name, a.name LIMIT 3")
ordered("A14", "MATCH (a:Person)-[r*1..2]->(b) RETURN DISTINCT b ORDER BY b.name")
ordered("A15", "MATCH (a:Person) RETURN a.name AS n, a ORDER BY n LIMIT 2")
ordered("A16", "MATCH (a:Person)-[r]->(b) RETURN r ORDER BY r.w DESC SKIP 7")
case("A17", "MATCH (a:Person)-[r]->(b) RETURN DISTINCT r SKIP 0 LIMIT 100")
case("A18", "MATCH (a:Person)-[r]->(b) RETURN r LIMIT 0")
case("A19", "MATCH (a:Person)-[r]->(b) RETURN r SKIP 100")

# B: nodes and edges through ORDER BY, LIMIT, SKIP and DISTINCT before RETURN, then a
# property read or a later pattern
case("B1", "MATCH (a:Person)-[r:KNOWS]->(b) WITH r ORDER BY r.since DESC LIMIT 2 RETURN r.w AS w, r.since AS s, type(r) AS t", CYPHER)
case("B2", "MATCH (a:Person) WITH a ORDER BY a.age SKIP 1 LIMIT 2 MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name", CYPHER)
case("B3", "MATCH (a:Person)-[r]->(b) WITH DISTINCT r RETURN r.w AS w", CYPHER)
case("B4", "MATCH (a:Person)-[r]->(b) WITH r, b ORDER BY r.w SKIP 2 RETURN r.w, b.name", CYPHER)
ordered("B5", "MATCH (a:Person)-[r]->(b) LET x = r RETURN x.w AS w ORDER BY w", GQL)
ordered("B6", "MATCH (a:Person)-[r]->(b) LET x = r RETURN x ORDER BY x.w", GQL)
case("B7", "MATCH (a:Person)-[r]->(b) WITH r AS e ORDER BY e.w LIMIT 3 RETURN e.w, type(e)", CYPHER)
case("B8", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY r.w LIMIT 3 MATCH ()-[r]->(c) RETURN r.w, c.name", CYPHER)
case("B9", "MATCH (a:Person) WITH a ORDER BY a.name LIMIT 1 MATCH (a)-[r]->(x) RETURN type(r) AS t, x.name AS n", CYPHER)
case("B10", "MATCH (a:Person)-[r]->(b) WITH r, a ORDER BY a.name RETURN startNode(r).name AS s, endNode(r).name AS e", CYPHER)
ordered("B11", "MATCH (a:Person)-[r]->(b) FILTER r.w > 2 RETURN r.w ORDER BY r.w", GQL)
case("B12", "MATCH (a:Person)-[r]->(b) WITH DISTINCT a RETURN a.w AS w, a.name AS n", CYPHER)
case("B13", "MATCH (a:Person)-[r]->(b) WITH r SKIP 1 RETURN r.w AS w, type(r) AS t", CYPHER)
case("B14", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY r.w DESC LIMIT 2 RETURN r", CYPHER)
case("B15", "MATCH (a:Person)-[r]->(b) WITH a, r ORDER BY r.w LIMIT 4 RETURN a.name, r.w, type(r)", CYPHER)
case("B16", "MATCH (a:Person)-[r]->(b) WITH collect(r) AS rs UNWIND rs AS x RETURN x.w AS w", CYPHER)
ordered("B17", "MATCH (a:Person)-[r]->(b) LET x = r LET y = x.w RETURN y ORDER BY y", GQL)

# C: set operations
case("C1", "MATCH (a:Person) RETURN a.name AS n UNION MATCH (c:City) RETURN c.name AS n")
case("C2", "MATCH (a:Person) RETURN a AS x UNION MATCH (c:City) RETURN c AS x")
case("C3", "MATCH ()-[r:KNOWS]->() RETURN r AS x UNION ALL MATCH ()-[r:LIVES_IN]->() RETURN r AS x")
case("C4", "MATCH (a:Person) RETURN a AS x UNION ALL MATCH ()-[r:LIVES_IN]->() RETURN r AS x")
case("C5", "MATCH ()-[x:KNOWS]->() RETURN x UNION ALL MATCH (x:City) RETURN x")
case("C6", "MATCH (a:Person) WHERE a.age > 30 RETURN a AS x UNION MATCH (c:City) RETURN c AS x UNION MATCH ()-[r:LIVES_IN]->() RETURN r AS x")
case("C7", "MATCH (a:Person)-[r*1..2]->(b) WHERE a.name = 'Jules' RETURN r UNION ALL MATCH ()-[r:LIVES_IN]->() RETURN r")
case("C8", "MATCH (a:Person)-[]->(b) RETURN a EXCEPT ALL MATCH (a:Person {name: 'Alix'}) RETURN a", GQL)
case("C9", "MATCH (a:Person)-[]->(b) RETURN a INTERSECT ALL MATCH (a:Person)-[:KNOWS]->(b) RETURN a", GQL)
case("C10", "UNWIND [1, 2, 2, 3, null] AS x RETURN x EXCEPT UNWIND [2, null] AS x RETURN x", GQL, "empty")
case("C11", "UNWIND [1, 2, 2, 3] AS x RETURN x INTERSECT ALL UNWIND [2, 2, 2, 4] AS x RETURN x", GQL, "empty")
case("C12", "MATCH (a:Person {name: 'Mia'}) RETURN a OTHERWISE MATCH (c:City) RETURN c AS a", GQL)
case("C13", "CALL { MATCH (a:Person) RETURN a AS x UNION MATCH (c:City) RETURN c AS x } RETURN x.name AS n", CYPHER)
case("C14", "MATCH (a:Person) WHERE a.age > 30 RETURN a.name AS n, a AS x UNION MATCH (a:Person) WHERE a.age < 30 RETURN a.name AS n, a AS x")
case("C16", "MATCH (a:Person)-[r]->(b) RETURN r, a UNION ALL MATCH (a:Person)-[r]->(b) RETURN r, b AS a")
case("C17", "MATCH (a:Person)-[:KNOWS]->(b) | (a:Person)-[:LIVES_IN]->(b) RETURN a.name, b.name", GQL)
case("C18", "MATCH (a:Person) WHERE a.name = 'Alix' RETURN a UNION ALL MATCH (a:Person) WHERE a.name = 'Alix' RETURN a")
case("C19", "MATCH (a:Person) WHERE a.name = 'Alix' RETURN a UNION MATCH (a:Person) WHERE a.name = 'Alix' RETURN a")
case("C20", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN a, r EXCEPT MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN a, r", GQL)
case("C21", "MATCH (a:Person) RETURN a.name AS n EXCEPT MATCH (a:Person) WHERE a.age > 29 RETURN a.name AS n", GQL)
case("C22", "MATCH (a:Person) WHERE a.age > 100 RETURN a.name AS n OTHERWISE MATCH (c:City) RETURN c.name AS n", GQL)
case("C23", "MATCH (a:Person)-[r]->(b) WHERE r.w < 3 RETURN r UNION MATCH (a:Person)-[r]->(b) WHERE r.w > 1 RETURN r")
case("C24", "MATCH (x:Person {name: 'Gus'}) RETURN x UNION ALL MATCH ()-[x:KNOWS]->() WHERE x.w = 1 RETURN x UNION ALL MATCH (x:City {name: 'Paris'}) RETURN x")
case("C25", "MATCH (a:Person) RETURN a.w AS w UNION ALL MATCH ()-[r:KNOWS]->() RETURN r.w AS w")

# D: sort keys that are not returned, and sort keys on aliases
ordered("D1", "MATCH (a:Person) RETURN a AS x ORDER BY x.age")
ordered("D2", "MATCH (a:Person) RETURN a.name AS n ORDER BY n")
ordered("D3", "MATCH (a:Person) RETURN a.name AS n ORDER BY a.age DESC")
ordered("D4", "MATCH (a:Person)-[r]->(b) RETURN a AS x, b AS y ORDER BY x.age, y.name")
ordered("D5", "MATCH (a:Person) RETURN DISTINCT a AS x ORDER BY x.name")
ordered("D6", "MATCH (a:Person) RETURN a AS x ORDER BY x.age SKIP 1 LIMIT 2")
ordered("D7", "MATCH (a:Person) RETURN a.age AS g, count(*) AS c ORDER BY g")
ordered("D8", "MATCH (a:Person)-[r]->(b) RETURN r AS e, b.name AS n ORDER BY e.w DESC, n")
ordered("D9", "MATCH (a:Person) RETURN a.name AS n ORDER BY toLower(n)")
ordered("D10", "MATCH (a:Person) RETURN a AS x ORDER BY labels(x)[0], x.name")
ordered("D11", "MATCH (a:Person)-[r]->(b) RETURN type(r) AS t ORDER BY r.w")
ordered("D12", "MATCH (a:Person) RETURN a.name ORDER BY a.age")
ordered("D13", "MATCH (a:Person) RETURN a ORDER BY a.age DESC LIMIT 1")
case("D14", "MATCH (a:Person) WITH a AS x ORDER BY x.age RETURN x.name AS n", CYPHER)
ordered("D15", "MATCH (a:Person) RETURN a.name AS n, a.age AS g ORDER BY g DESC, n")
ordered("D16", "MATCH (a:Person) RETURN a AS x ORDER BY x.age DESC, x.name LIMIT 3")
ordered("D17", "MATCH (a:Person)-[r]->(b) RETURN r AS e ORDER BY e.w, e.since LIMIT 4")
ordered("D18", "MATCH (a:Person) RETURN a AS x, a.name AS n ORDER BY x.age")
ordered("D19", "MATCH (a:Person) RETURN a AS x ORDER BY x.w")
ordered("D20", "MATCH (a:Person)-[r]->(b) RETURN DISTINCT r AS e ORDER BY e.w LIMIT 2")
ordered("D21", "MATCH (a:Person) RETURN a AS x ORDER BY x.age + 1")
ordered("D22", "MATCH (a:Person) RETURN a.name AS n, a AS x ORDER BY x.age SKIP 2")

# E: properties and ids read after a node or edge went through ORDER BY, LIMIT, SKIP
# or DISTINCT
for case_id, query in [
    ("E1", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY type(r), r.w LIMIT 3 RETURN r.w AS w"),
    ("E2", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY toString(r.w) LIMIT 3 RETURN r.w AS w"),
    ("E3", "MATCH (a:Person) WITH a ORDER BY a.age DESC LIMIT 2 RETURN a.w AS w, a.name AS n"),
    ("E4", "MATCH (a:Person)-[r]->(b) WITH a, r ORDER BY toString(r.w) RETURN r.w AS w, a.name AS n"),
    ("E5", "MATCH (a:Person) WITH a ORDER BY a.name SKIP 1 RETURN a.w AS w, a.name AS n"),
    ("E6", "MATCH p = (a:Person)-[:KNOWS]->(b) WITH p, a ORDER BY a.name LIMIT 2 RETURN [n IN nodes(p) | n.w] AS ws"),
    ("E7", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY r.w LIMIT 3 RETURN id(r) AS i, r.w AS w"),
    ("E8", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY r.w LIMIT 2 RETURN properties(r) AS p"),
    ("E9", "MATCH (a:Person)-[r:KNOWS]->(b) WITH r ORDER BY r.w DESC RETURN r.since AS s, r.w AS w"),
    ("E10", "MATCH (a:Person)-[r]->(b) WITH DISTINCT r, b ORDER BY b.name LIMIT 4 RETURN r.w AS w, b.w AS bw"),
    ("E11", "MATCH (a:Person)-[r]->(b) WITH r, a ORDER BY a.age, r.w SKIP 1 LIMIT 3 RETURN r.w AS w, a.w AS aw"),
    ("E12", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY r.w LIMIT 3 MATCH (x)-[r]->(y) RETURN x.w AS xw, y.name AS y"),
    ("E13", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY labels(startNode(r))[0], r.w LIMIT 2 RETURN r.w AS w"),
    ("E14", "MATCH (a:Person)-[r]->(b) WITH b ORDER BY b.name LIMIT 3 RETURN b.w AS w, b.name AS n"),
    ("E15", "MATCH (a:Person)-[r]->(b) WITH r ORDER BY r.w DESC SKIP 5 RETURN keys(r) AS k, r.w AS w"),
]:
    case(case_id, query, CYPHER)
ordered("E16", "MATCH (a:Person)-[r]->(b) RETURN r.w AS w, a.w AS aw ORDER BY r.w SKIP 2 LIMIT 3")
ordered("E17", "MATCH (a:Person)-[r]->(b) RETURN DISTINCT a.w AS aw, b.w AS bw ORDER BY aw, bw")

# F: cuts across row batches
ordered("F1", "MATCH (n:N) RETURN n.i ORDER BY n.i SKIP 2046 LIMIT 5", fixture="chain")
ordered("F2", "MATCH (n:N) RETURN n.i ORDER BY n.i DESC LIMIT 3", fixture="chain")
ordered("F3", "MATCH (n:N) RETURN n ORDER BY n.i SKIP 4998", fixture="chain")
ordered("F4", "MATCH (n:N)-[r]->(m) RETURN r ORDER BY r.k SKIP 2047 LIMIT 3", fixture="chain")
ordered("F5", "MATCH (n:N) RETURN DISTINCT n.m AS m ORDER BY m", fixture="chain")
ordered("F6", "MATCH (n:N)-[r]->(m) WHERE r.k % 1000 = 0 RETURN r ORDER BY r.k", fixture="chain")
case("F7", "MATCH (n:N) RETURN count(*) AS c", fixture="chain")
case("F8", "MATCH (n:N) WITH n ORDER BY n.i DESC LIMIT 3 MATCH (n)<-[r]-(p) RETURN n.i, r.k, p.i", CYPHER, "chain")
case("F9", "UNWIND range(1, 5000) AS i RETURN i SKIP 2047 LIMIT 3", fixture="empty")
ordered("F10", "MATCH (n:N) RETURN n.i AS i ORDER BY i LIMIT 2050", fixture="chain")
ordered("F11", "MATCH (n:N)-[r]->(m) RETURN DISTINCT r ORDER BY r.k LIMIT 2", fixture="chain")
case("F12", "MATCH (n:N) RETURN n SKIP 2047 LIMIT 2", fixture="chain")
case("F13", "MATCH (n:N)-[r]->(m) RETURN r SKIP 4997", fixture="chain")
case("F14", "MATCH (n:N)-[r]->(m) WITH r SKIP 4990 RETURN r.k AS k", CYPHER, "chain")
case("F15", "MATCH (n:N) WITH n ORDER BY n.i SKIP 2045 LIMIT 5 RETURN n.i AS i, n.m AS m", CYPHER, "chain")
ordered("F16", "MATCH (n:N) RETURN n.m AS m, count(*) AS c ORDER BY m", fixture="chain")

# G: values of different types in one column
ordered("G1", "UNWIND [3, 'a', 2.5, null, true, [1, 2], {k: 1}] AS x RETURN x ORDER BY x", fixture="empty")
ordered("G2", "UNWIND [3, 'a', 2.5, null, true, [1, 2], {k: 1}] AS x RETURN x ORDER BY x DESC LIMIT 3", fixture="empty")
case("G3", "UNWIND [1, 1, 1.0, '1', null, null, true] AS x RETURN DISTINCT x", fixture="empty")
case("G4", "UNWIND [1, 'a', 2.5, null] AS x RETURN x SKIP 1 LIMIT 2", fixture="empty")
case("G5", "UNWIND [1, 'a', null] AS x RETURN x EXCEPT UNWIND ['a'] AS x RETURN x", GQL, "empty")
ordered("G6", "UNWIND range(1, 3000) AS i RETURN CASE WHEN i % 2 = 0 THEN i ELSE toString(i) END AS v ORDER BY i LIMIT 3", fixture="empty")
case("G7", "UNWIND range(1, 3000) AS i WITH CASE WHEN i < 2500 THEN i ELSE 'x' END AS v RETURN v SKIP 2498 LIMIT 3", CYPHER, "empty")
ordered("G8", "UNWIND range(1, 3000) AS i RETURN CASE WHEN i > 2048 THEN 'late' ELSE i END AS v ORDER BY i DESC LIMIT 2", fixture="empty")
ordered("G9", "UNWIND [date('2024-01-02'), date('2023-05-06')] AS d RETURN d ORDER BY d", fixture="empty")
ordered("G10", "UNWIND [1, 2, 3] AS x RETURN x, CASE WHEN x = 2 THEN 'two' ELSE x END AS y ORDER BY x DESC SKIP 1", fixture="empty")
case("G11", "UNWIND range(1, 3000) AS i RETURN DISTINCT CASE WHEN i > 2048 THEN 'late' ELSE i % 3 END AS v", fixture="empty")
ordered("G12", "UNWIND range(1, 5) AS i RETURN i, i * 1.5 AS f ORDER BY f DESC LIMIT 2", fixture="empty")

# H: GQL FOR and LET with ordering
ordered("H1", "FOR x IN [3, 1, 2] RETURN x ORDER BY x LIMIT 2", GQL, "empty")
ordered("H2", "MATCH (a:Person) LET n = a.name RETURN n ORDER BY n SKIP 1 LIMIT 2", GQL)
ordered("H3", "MATCH (a:Person) LET b = a RETURN b ORDER BY b.age LIMIT 2", GQL)

# I: RETURN * with ORDER BY
ordered("I1", "MATCH (a:Person)-[r:LIVES_IN]->(c) RETURN * ORDER BY r.years")
ordered("I2", "MATCH (a:Person)-[r:LIVES_IN]->(c) RETURN * ORDER BY a.name LIMIT 1")

# J: aggregation over nodes and edges
ordered("J1", "MATCH (a:Person)-[r]->(b) RETURN b.name AS n, count(r) AS c ORDER BY c DESC, n")
case("J2", "MATCH (a:Person)-[r]->(b) RETURN r, count(*) AS c")
case("J3", "MATCH (a:Person)-[r]->(b) RETURN collect(r.w) AS ws")

# K: variables bound before a later pattern, and values through a later MATCH
for case_id, query, languages in [
    ("K1", "MATCH ()-[r]->() MATCH (x)-[r]->(y) RETURN r.w AS w, x.name AS x, y.name AS y", BOTH),
    ("K2", "MATCH (a)-[r]->(b) MATCH (a)-[r]->(c) RETURN r.w AS w, c.name AS c", BOTH),
    ("K3", "MATCH (a)-[r]->(b) MATCH (x)<-[r]-(y) RETURN r.w AS w, x.name AS x, y.name AS y", BOTH),
    ("K4", "MATCH (a)-[r]->(b) MATCH (x)-[r]-(y) RETURN r.w AS w, x.name AS x", BOTH),
    ("K5", "MATCH ()-[r:KNOWS]->() MATCH ()-[r:LIVES_IN]->() RETURN count(*) AS n", BOTH),
    ("K6", "MATCH ()-[r]->() MATCH (x)-[r {w: 3}]->(y) RETURN x.name AS x", BOTH),
    ("K7", "MATCH ()-[r]->() CALL { WITH r MATCH (x)-[r]->(y) RETURN y.name AS t } RETURN r.w AS w, t", BOTH),
    ("K8", "MATCH ()-[r]->() CALL { WITH * MATCH (x)-[r]->(y) RETURN y.name AS t } RETURN r.w AS w, t", BOTH),
    ("K9", "MATCH (a)-->(b) CALL { WITH a, b MATCH (b)-->(a) RETURN count(*) AS c } RETURN a.name AS a, b.name AS b, c", BOTH),
    ("K10", "MATCH (a:Person) MATCH (c:City) RETURN a.name AS n, a.w AS w, c.name AS c", BOTH),
    ("K11", "MATCH ()-[r:KNOWS]->() MATCH (c:City) RETURN r.w AS w, c.name AS c", BOTH),
    ("K12", "MATCH ()-[r:KNOWS]->() MATCH (c:City) WHERE r.w > 2 RETURN r.w AS w, c.name AS c", BOTH),
    ("K13", "MATCH (a:Person) MATCH (a)-[:LIVES_IN]->(c) RETURN a.w AS w, c.w AS cw", BOTH),
    ("K14", "MATCH (a:Person) WITH a MATCH (b:Person) WHERE a.age < b.age RETURN a.name AS a, b.name AS b", BOTH),
    ("K15", "MATCH (a:Person)-[r:KNOWS]->(b) MATCH (c:City) RETURN a.w AS aw, r.w AS rw, b.w AS bw, c.w AS cw", BOTH),
    ("K16", "MATCH (a:Person) MATCH (b:Person) RETURN count(*) AS n", BOTH),
    ("K17", "MATCH (a:Person) MATCH (c:City) RETURN a, c ORDER BY a.name, c.name LIMIT 4", BOTH),
    ("K18", "MATCH ()-[r:KNOWS]->() MATCH (c:City) RETURN r ORDER BY r.w, c.name LIMIT 3", BOTH),
    ("K19", "MATCH (a:Person) MATCH p = (a)-[:KNOWS*1..2]->(b) RETURN a.name AS a, length(p) AS l, b.w AS bw", BOTH),
    ("K20", "MATCH ()-[r:KNOWS]->() MATCH p = (x:Person {name: 'Mia'})-->(y) RETURN r.w AS w, y.name AS y", BOTH),
    ("K21", "MATCH (a:Person) MATCH (c:City) RETURN sum(a.w) AS s, count(a.w) AS n", BOTH),
    ("K22", "UNWIND [1, 2] AS k MATCH (a:Person {name: 'Jules'}) RETURN k, a.w AS w", BOTH),
    ("K23", "MATCH (a:Person {name: 'Jules'}) WITH a MATCH (c:City {name: 'Paris'}) RETURN a.w AS w, c.w AS cw", BOTH),
    ("K24", "MATCH (a:Person)-[r]->(b) MATCH (a)-[s]->(c) WHERE r <> s RETURN r.w AS rw, s.w AS sw", BOTH),
    ("K25", "MATCH (a)-[r*1..2]->(b) MATCH (x)-[r*1..2]->(y) RETURN count(*) AS n", BOTH),
    ("K26", "MATCH (a)-[r]->(b)-[r]->(c) RETURN count(*) AS n", BOTH),
    ("K27", "MATCH ()-[r]->() OPTIONAL MATCH (x)-[r:KNOWS]->(y) RETURN r.w AS w, y.name AS y", BOTH),
    ("K28", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c) RETURN a.name AS n, c.name AS c", BOTH),
    ("K29", "MATCH (a:Person) MATCH (b:Person) WHERE id(a) < id(b) RETURN count(*) AS n", BOTH),
    ("K30", "MATCH (a:Person)-[r:KNOWS]->(b) MATCH (b)-[s:LIVES_IN]->(c) RETURN a.name AS a, r.w AS rw, s.w AS sw, c.w AS cw", BOTH),
    ("K31", "MATCH (a:Person) MATCH (c:City) WITH a, c ORDER BY a.name, c.name LIMIT 3 RETURN a.w AS w, c.w AS cw", CYPHER),
    ("K32", "MATCH (a:Person) MATCH (c:City) RETURN DISTINCT a.w AS w", BOTH),
]:
    case(case_id, query, languages)
case("K40", "MATCH (a:N) WHERE a.i < 3 MATCH (b:N) WHERE b.i >= 4990 RETURN a.i AS ai, b.i AS bi, b.k AS bk", fixture="chain")
case("K41", "MATCH (a:N) WHERE a.i < 1 MATCH (b:N) RETURN count(*) AS n, sum(b.i) AS s", fixture="chain")
case("K42", "MATCH ()-[r:NEXT]->() WHERE r.k < 3 MATCH (b:N) WHERE b.i < 2 RETURN r.k AS k, r.i AS ri, b.i AS i", fixture="chain")
case("K43", "MATCH ()-[r:NEXT]->() WHERE r.k < 2100 MATCH (b:N {i: 7}) RETURN count(r.k) AS n, sum(r.k) AS s, count(r.i) AS ri", fixture="chain")

# L: ORDER BY across types, NaN and nulls, top-K and percentiles
for case_id, query, languages, fixture in [
    ("L1", "UNWIND [3, 'a', 2.5, true, null, [1], {k: 1}] AS x RETURN x ORDER BY x", BOTH, "social"),
    ("L2", "UNWIND [3, 'a', 2.5, true, null, [1], {k: 1}] AS x RETURN x ORDER BY x DESC", BOTH, "social"),
    ("L3", "MATCH (n) RETURN n.w AS w ORDER BY w", BOTH, "social"),
    ("L4", "MATCH (n) RETURN n.w AS w ORDER BY w DESC", BOTH, "social"),
    ("L5", "MATCH (n) RETURN n.w AS w ORDER BY w DESC NULLS LAST", GQL, "social"),
    ("L6", "MATCH (n) RETURN n.w AS w ORDER BY w ASC NULLS FIRST", GQL, "social"),
    ("L7", "MATCH (n) RETURN n.name AS n2, n.w AS w ORDER BY w DESC NULLS LAST, n2 LIMIT 3", GQL, "social"),
    ("L8", "MATCH (a:Person) RETURN a.name AS n ORDER BY a.age DESC", BOTH, "social"),
    ("L9", "MATCH (a:Person)-[r]->(b) RETURN type(r) AS t, r.w AS w ORDER BY t DESC, w", BOTH, "social"),
    ("L10", "UNWIND [1, 0.0 / 0.0, -1, 1.0 / 0.0] AS x RETURN x ORDER BY x", BOTH, "social"),
    ("L11", "MATCH (n) RETURN labels(n) AS l, n.name AS n2 ORDER BY l, n2", BOTH, "social"),
    ("L12", "MATCH (n) WITH n ORDER BY n.w DESC LIMIT 3 RETURN n.name AS n2", CYPHER, "social"),
    ("L13", "MATCH (n) RETURN n.w AS w ORDER BY w DESC LIMIT 2", BOTH, "social"),
    ("L14", "MATCH (n:N) RETURN n.i AS i ORDER BY i DESC LIMIT 3", BOTH, "chain"),
    ("L15", "MATCH (n:N) RETURN n.i AS i, n.m AS m ORDER BY m DESC, i LIMIT 5", BOTH, "chain"),
    ("L16", "MATCH (n:N) WHERE n.i < 10 RETURN percentileCont(n.i, 0.5) AS p, percentileDisc(n.i, 0.5) AS d", BOTH, "chain"),
    ("L17", "MATCH ()-[r:NEXT]->() RETURN r.k AS k ORDER BY k DESC SKIP 10 LIMIT 3", BOTH, "chain"),
    ("L18", "UNWIND range(1, 3000) AS i RETURN CASE WHEN i % 3 = 0 THEN toString(i) ELSE i END AS v ORDER BY v LIMIT 3", BOTH, "social"),
    ("L19", "MATCH (n) RETURN n.name AS n2 ORDER BY n.w DESC, n2 LIMIT 4", BOTH, "social"),
    ("L20", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c) RETURN a.name AS n, c.name AS c ORDER BY c DESC, n", BOTH, "social"),
]:
    ordered(case_id, query, languages, fixture)

# M: EXISTS and COUNT subqueries: nodes and edges shared with the row, nulls, paths, a
# second pattern and path modes
for case_id, query, languages, fixture in [
    ("M1", "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE EXISTS { MATCH (b)-[:KNOWS]->(a) } RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("M2", "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE NOT EXISTS { MATCH (b)-[:KNOWS]->(a) } RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("M3", "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE EXISTS { MATCH (b)-[:KNOWS]->(a) } OR a.name = 'Jules' RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("M4", "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE EXISTS { MATCH (c)-[:KNOWS]->(a) } RETURN a.name AS a, b.name AS b, c.name AS c", BOTH, "social"),
    ("M5", "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.name AS a, c.name AS c, EXISTS { MATCH (c)-[:KNOWS]->(a) } AS closes, COUNT { MATCH (c)-[:KNOWS]->(a) } AS n", BOTH, "social"),
    ("M6", "MATCH (a)-[r]->(b) WHERE EXISTS { MATCH (x)-[r]->(:City) } RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("M7", "MATCH (a)-[r]->(b) RETURN a.name AS a, b.name AS b, EXISTS { MATCH (x)-[r]->(:City) } AS home, COUNT { MATCH ()-[r]->() } AS one", BOTH, "social"),
    ("M8", "MATCH (c:City) WHERE EXISTS { MATCH (x)-[:WORKS_AT]->() } RETURN c.name AS c", BOTH, "social"),
    ("M9", "MATCH (c:City) WHERE NOT EXISTS { MATCH (x)-[:WORKS_AT]->() } RETURN c.name AS c", BOTH, "social"),
    ("M10", "MATCH (c:City) RETURN c.name AS c, COUNT { MATCH (x)-[:KNOWS]->(y) } AS knows, EXISTS { MATCH (x)-[:LIVES_IN]->(y) } AS lives", BOTH, "social"),
    ("M11", "MATCH (c:City) RETURN c.name AS c, EXISTS { MATCH (p)-[:LIVES_IN]->(c) } AS e, COUNT { MATCH (p)-[:LIVES_IN]->(c) } AS n", BOTH, "social"),
    ("M12", "MATCH (c:City) WHERE EXISTS { MATCH (p)-[:LIVES_IN]->(c) } RETURN c.name AS c", BOTH, "social"),
    ("M13", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c) RETURN a.name AS a, EXISTS { MATCH (c)<-[:LIVES_IN]-() } AS e, COUNT { MATCH (c)<-[:LIVES_IN]-() } AS n", BOTH, "social"),
    ("M14", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c) WITH a, c WHERE NOT EXISTS { MATCH (c)<-[:LIVES_IN]-() } RETURN a.name AS a", BOTH, "social"),
    ("M15", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c) WITH a, c WHERE EXISTS { MATCH (c)<-[:LIVES_IN]-() } RETURN a.name AS a", BOTH, "social"),
    ("M16", "MATCH (a:Person {name: 'Alix'}), (b:Person) RETURN b.name AS b, EXISTS { MATCH (a)-[:KNOWS*1..2]->(b) } AS near", BOTH, "social"),
    ("M17", "MATCH (a:Person {name: 'Alix'}), (b:Person) WHERE EXISTS { MATCH (a)-[:KNOWS*1..2]->(b) } RETURN b.name AS b", BOTH, "social"),
    ("M18", "MATCH (a:Person {name: 'Alix'}), (b:Person) RETURN b.name AS b, EXISTS { MATCH (a)-[:KNOWS*]->(b) } AS reach", BOTH, "social"),
    ("M19", "MATCH (a:Person) WITH a, a AS b RETURN a.name AS a, EXISTS { MATCH (a)-[:KNOWS*1..2]-(b) } AS back, EXISTS { MATCH (a)-[:KNOWS*1..3]->(b) } AS closed", BOTH, "social"),
    ("M20", "MATCH (a:Person {name: 'Alix'})-[rs:KNOWS*1..3]->(b) RETURN b.name AS b, size(rs) AS hops, EXISTS { MATCH (x)<-[rs:KNOWS*]-(y) } AS rev, EXISTS { MATCH (x)-[rs:KNOWS*1..2]->(y) } AS short, EXISTS { MATCH (b)-[rs:KNOWS*]->(y) } AS from_b", BOTH, "social"),
    ("M21", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b), (c:Robot) } RETURN a.name AS a", BOTH, "social"),
    ("M22", "MATCH (a:Person) RETURN a.name AS a, EXISTS { MATCH (a)-[:KNOWS]->(b), (c:Robot) } AS e", BOTH, "social"),
    ("M23", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b) MATCH (b:City) } RETURN a.name AS a", CYPHER, "social"),
    ("M24", "MATCH (a:Person) RETURN a.name AS a, COUNT { MATCH (a)-[r]->(b) MATCH (b:City) } AS cities", CYPHER, "social"),
    ("M25", "MATCH (a:Person) WHERE EXISTS { MATCH ACYCLIC (a)-[:KNOWS*1..3]->(a) } RETURN a.name AS a", GQL, "social"),
    ("M26", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS*1..3]->(a) } RETURN a.name AS a", GQL, "social"),
    ("M27", "MATCH (a:Person) RETURN a.name AS a, EXISTS { MATCH TRAIL (a)-[:KNOWS*1..2]->(x) } AS e", GQL, "social"),
    ("M28", "MATCH (a:Person) RETURN a.name AS a, CASE WHEN EXISTS { MATCH (a)-[:LIVES_IN]->() } THEN 'housed' ELSE 'not' END AS h", BOTH, "social"),
    ("M29", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:LIVES_IN]->(:City) } OR a.age > 35 RETURN a.name AS a", BOTH, "social"),
    ("M30", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:LIVES_IN]->() } AND EXISTS { MATCH (a)-[:KNOWS]->()-[:KNOWS]->() } RETURN a.name AS a", BOTH, "social"),
    ("M31", "MATCH (a:Person), (b:Person) WHERE (a)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b", CYPHER, "social"),
    ("M32", "MATCH (a:Person), (b:Person) WHERE NOT (a)-[:KNOWS]->(b) AND a <> b RETURN a.name AS a, b.name AS b", CYPHER, "social"),
    ("M33", "MATCH (a:Person), (c:City) WHERE EXISTS { MATCH (a)-[:LIVES_IN]->(c:City) } RETURN a.name AS a, c.name AS c", BOTH, "social"),
    ("M34", "MATCH (a:Person) RETURN a.name AS a, COUNT { MATCH (a)-[:KNOWS]->(f) } AS out, COUNT { MATCH (a)<-[:KNOWS]-(f) } AS inn, COUNT { MATCH (a)-[:KNOWS]-(f) } AS both", BOTH, "social"),
    ("M35", "MATCH (a:Person) WHERE COUNT { MATCH (a)-[:KNOWS]->(f) } = 2 RETURN a.name AS a", BOTH, "social"),
    ("M36", "MATCH (a:Person)-[:KNOWS]->(b) WITH a, b WHERE COUNT { MATCH (b)-[:KNOWS]->(a) } = 0 RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("M37", "MATCH (a:N)-[:NEXT]->(b:N) WHERE EXISTS { MATCH (b)-[:NEXT]->(a) } RETURN count(*) AS n", BOTH, "chain"),
    ("M38", "MATCH (a:N)-[:NEXT]->(b:N) WHERE NOT EXISTS { MATCH (b)-[:NEXT]->(a) } RETURN count(*) AS n", BOTH, "chain"),
    ("M39", "MATCH (n:N) WHERE n.i < 4 RETURN n.i AS i, EXISTS { MATCH (n)-[:NEXT]->(m) } AS out, COUNT { MATCH (m)-[:NEXT]->(n) } AS inn ORDER BY i", BOTH, "chain"),
    ("M40", "MATCH (a:N {i: 10})-[:NEXT]->(b) RETURN b.i AS i, EXISTS { MATCH (a)-[:NEXT]->(b) } AS e, COUNT { MATCH (b)<-[:NEXT]-(a) } AS c", BOTH, "chain"),
]:
    case(case_id, query, languages, fixture)

# fmt: on
