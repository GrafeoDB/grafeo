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


def labels(grafeo, indexed=False):
    """Skewed labels: 1,988 `Graph` nodes (`g0` to `g1987`), three nodes with `Graph`
    and `Repository` (`r0` to `r2`) and 19 `Repository` nodes (`p0` to `p18`), each with
    `id` and `n` (its number); every `r<i>` HAS `g<i>`. `Tag` and `Topic` have three nodes
    each, `t0` has both. With `indexed`, a property index on `id`."""
    db = grafeo.GrafeoDB()
    if indexed:
        db.create_property_index("id")
    for label, prefix, count in [
        ("Graph", "g", 1988),
        ("Graph:Repository", "r", 3),
        ("Repository", "p", 19),
    ]:
        db.execute(
            f"UNWIND range(0, {count - 1}) AS i "
            f"INSERT (:{label} {{id: '{prefix}' + toString(i), n: i, w: 100 + i}})"
        )
    db.execute(
        "MATCH (r:Repository), (g:Graph) WHERE r.id STARTS WITH 'r' "
        "AND g.id = 'g' + toString(r.n) INSERT (r)-[:HAS {w: r.n}]->(g)"
    )
    db.execute("INSERT (:Tag:Topic {id: 't0'}), (:Tag {id: 't1'}), (:Tag {id: 't2'})")
    db.execute("INSERT (:Topic {id: 'u1'}), (:Topic {id: 'u2'})")
    return db


def labels_indexed(grafeo):
    """`labels` with a property index on `id`."""
    return labels(grafeo, indexed=True)


FIXTURES = {
    "social": social,
    "chain": chain,
    "empty": empty,
    "labels": labels,
    "labels_indexed": labels_indexed,
}

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

# N: nodes and edges through joins, EXISTS, CALL, UNWIND and group keys
for case_id, query in [
    ("N1", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c) RETURN a.name AS a, a.w AS w, c.w AS cw"),
    ("N2", "MATCH (a:Person)-[:KNOWS]->(b), (b)-[r:LIVES_IN]->(c) RETURN a.name AS a, b.name AS b, b.w AS bw, r.w AS rw"),
    ("N3", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(m)-[:LIVES_IN]->() } RETURN a.name AS a, a.w AS w"),
    ("N4", "MATCH (a:Person) WHERE NOT EXISTS { MATCH (a)-[:KNOWS]->()-[:KNOWS]->() } RETURN a.name AS a, a.w AS w"),
    ("N5", "MATCH (a:Person) CALL { WITH a RETURN 1 AS one } RETURN a.name AS a, a.w AS w"),
    ("N6", "MATCH ()-[r:KNOWS]->() CALL { RETURN 1 AS one } RETURN r.w AS w"),
    ("N7", "MATCH (a:Person) CALL { WITH a MATCH (a)-[r:KNOWS]->(b) RETURN r, b } RETURN a.name AS a, r.w AS rw, b.w AS bw"),
    ("N8", "MATCH (a:Person) UNWIND [1, 2] AS k RETURN a.name AS a, a.w AS w, k"),
    ("N9", "MATCH (a:Person)-[r]->() WITH a, count(r) AS n RETURN a.name AS a, a.w AS w, n"),
    ("N10", "MATCH ()-[r:KNOWS]->() WITH r, count(*) AS n RETURN r.w AS w, n"),
    ("N11", "MATCH (c:City) OPTIONAL MATCH (p:Person)-[:LIVES_IN]->(c) RETURN c.name AS c, c.w AS w, p.name AS p"),
]:
    case(case_id, query)

# O: lists of nodes and edges (collect, path functions) and UNWIND of lists computed per row
for case_id, query in [
    ("O1", "MATCH (a:Person) WITH collect(a) AS people UNWIND people AS p RETURN p.name AS n, p.w AS w"),
    ("O2", "MATCH ()-[r:KNOWS]->() WITH collect(r) AS rs UNWIND rs AS e RETURN e.w AS w"),
    ("O3", "MATCH ()-[r:LIVES_IN]->() WITH collect(r) AS rs UNWIND rs AS e RETURN type(e) AS t, e.w AS w"),
    ("O4", "MATCH ()-[r:KNOWS]->() WITH collect(r) AS rs RETURN size([x IN rs WHERE x.w < 10]) AS n"),
    ("O5", "MATCH p = (:Person {name: 'Alix'})-[:KNOWS*2]->() UNWIND relationships(p) AS e RETURN e.w AS w"),
    ("O6", "MATCH p = (:Person {name: 'Jules'})-[:KNOWS]->() UNWIND nodes(p) AS n RETURN n.name AS n, n.w AS w"),
    ("O7", "MATCH (a:Person) UNWIND range(1, a.w - 99) AS i RETURN a.name AS a, i"),
    ("O8", "MATCH ()-[r:LIVES_IN {w: 6}]->() RETURN collect(r) AS rs"),
    ("O9", "MATCH (a:Person)-[r:KNOWS]->() WITH a, count(r) AS n RETURN a, n"),
    ("O10", "MATCH ()-[r:LIVES_IN]->() WITH collect(r) AS rs UNWIND rs AS e MATCH (x)-[e]->(y) RETURN x.name AS x, y.name AS y"),
    ("O11", "MATCH ()-[r:KNOWS {w: 5}]->() WITH collect(r) AS rs RETURN rs[0].w AS w, rs[-1].since AS since"),
]:
    case(case_id, query)

# P: keys() of nodes and edges, and a variable that is a node, an edge or a value
for case_id, query, languages in [
    ("P1", "MATCH ()-[r:KNOWS {w: 1}]->() RETURN size(keys(r)) AS n, 'since' IN keys(r) AS s", BOTH),
    ("P2", "MATCH ()-[r]->() WHERE 'since' IN keys(r) RETURN count(*) AS n", BOTH),
    ("P3", "MATCH (r) MATCH ()-[r]->() RETURN count(*) AS c", BOTH),
    ("P4", "MATCH (a)-[a]->(b) RETURN count(*) AS c", BOTH),
    ("P5", "MATCH (a)-[r]->(r) RETURN count(*) AS c", BOTH),
    ("P6", "WITH 1 AS r MATCH ()-[r]->() RETURN count(*) AS c", CYPHER),
    ("P7", "MATCH (a:Person) CALL { WITH a MATCH ()-[a]->(b) RETURN b } RETURN count(*) AS c", BOTH),
    ("P8", "MATCH ()-[r:KNOWS]->() CALL { WITH r MATCH (x)-[r]->(y) RETURN x.name AS xn } RETURN count(*) AS c", BOTH),
]:
    case(case_id, query, languages)

# Q: EXISTS and COUNT subqueries tied to the outer row by a value, or that one edge does not decide
for case_id, query in [
    ("Q1", "MATCH (a:Person) RETURN a.name AS n, EXISTS { MATCH (c:City) WHERE c.w = a.w + 4 } AS e"),
    ("Q2", "MATCH (a:Person) RETURN a.name AS n, COUNT { MATCH (b:Person) WHERE b.age < a.age } AS younger"),
    ("Q3", "MATCH (a:Person) WHERE COUNT { MATCH (b:Person) WHERE b.age < a.age } = 2 RETURN a.name AS n"),
    ("Q4", "UNWIND [25, 30] AS k RETURN k, COUNT { MATCH (p:Person {age: k}) } AS c"),
    ("Q5", "MATCH (a:Person) RETURN a.name AS n, COUNT { MATCH (a)-[:KNOWS*1..2]->() } AS c"),
    ("Q6", "MATCH (a:Person) WHERE COUNT { MATCH (a)-[:KNOWS]->(b), (c:Robot) } = 0 RETURN a.name AS n"),
    ("Q7", "MATCH (a:Person)-[:KNOWS]->(b) WITH a WHERE COUNT { MATCH (a)-[:KNOWS]->(x), (y:City) } = 6 RETURN a.name AS n"),
    ("Q8", "MATCH (a:Person) WITH a, COUNT { MATCH (b:Person) WHERE b.age < a.age } AS y RETURN a.name AS n, y"),
]:
    case(case_id, query)

# R: EXISTS in WHERE tied to the outer row by a value, inside OR, and through the row's edge;
#    R6: a CALL subquery over two edges in a row, which runs again for each outer row like those
for case_id, query in [
    ("R1", "MATCH (a:Person) WHERE EXISTS { MATCH (c:City) WHERE c.w = a.w + 4 } RETURN a.name AS n"),
    ("R2", "MATCH (a:Person) WHERE NOT EXISTS { MATCH (c:City) WHERE c.w = a.w + 4 } RETURN a.name AS n"),
    ("R3", "MATCH (a:Person)-[:KNOWS]->(b) WITH a WHERE a.name = 'Jules' OR EXISTS { MATCH (a)-[:KNOWS]->(x), (c:City) WHERE c.w = a.w + 4 } RETURN a.name AS n"),
    ("R4", "MATCH (a:Person)-[r:KNOWS]->(b) WHERE EXISTS { MATCH (a)-[s:KNOWS]->(c) WHERE s <> r } RETURN a.name AS a, b.name AS b"),
    ("R5", "UNWIND [25, 31] AS k WITH k WHERE EXISTS { MATCH (p:Person {age: k}) } RETURN k"),
]:
    case(case_id, query)
case("R6", "MATCH (a:Person) CALL { WITH a MATCH (a)-[:KNOWS]->(m)-[:KNOWS]->(x) RETURN x.name AS x } RETURN a.name AS a, x", CYPHER)

# S: GQL VALUE subqueries returning count(x) skip nulls and count DISTINCT values once
for case_id, query in [
    ("S1", "MATCH (a:Person) RETURN a.name AS n, VALUE { MATCH (a)-[:KNOWS]->(b) RETURN count(b.w) } AS c"),
    ("S2", "MATCH (a:Person) RETURN a.name AS n, VALUE { MATCH (a)-[:KNOWS]->(b), (c:City) RETURN count(DISTINCT b) } AS c"),
    ("S3", "MATCH (a:Person) WHERE VALUE { MATCH (a)-[:KNOWS]->(b) RETURN count(b.w) } = 1 RETURN a.name AS n"),
]:
    case(case_id, query, GQL)

# T: the MATCH clauses of a subquery go on from each other
for case_id, query, languages in [
    ("T1", "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b) MATCH (b)-[:LIVES_IN]->(c:City) } RETURN a.name AS n", BOTH),
    ("T2", "MATCH (a:Person) RETURN a.name AS n, COUNT { MATCH (a)-[:KNOWS]->(b) MATCH (b:City) } AS c", BOTH),
    ("T3", "MATCH (a:Person) RETURN a.name AS n, COUNT { MATCH (a)-[:KNOWS]->(b) OPTIONAL MATCH (b)-[:LIVES_IN]->(c) } AS c", GQL),
]:
    case(case_id, query, languages)

# U: a Cypher importing WITH only lists outer variables, as in Neo4j
for case_id, query in [
    ("U1", "MATCH (a:Person) CALL { WITH a WHERE a.age > 30 RETURN a.name AS m } RETURN m"),
    ("U2", "MATCH (a:Person) CALL { WITH a AS b RETURN b.name AS m } RETURN m"),
    ("U3", "MATCH (a:Person) CALL { WITH a WITH a WHERE a.age > 30 RETURN a.name AS m } RETURN m"),
    ("U4", "MATCH (a:Person) CALL { WITH a WITH a AS b RETURN b.name AS m } RETURN a.name AS a, m"),
]:
    case(case_id, query, CYPHER)

# V: the nodes and edges a CALL subquery returns are the nodes and edges themselves
V_NODE = "MATCH (a:Person {name: 'Alix'}) CALL { WITH a MATCH (a)-[:KNOWS]->(b) RETURN b }"
V_EDGE = "MATCH (a:Person {name: 'Alix'}) CALL { WITH a MATCH (a)-[r:KNOWS]->() RETURN r }"
for case_id, query in [
    ("V1", f"{V_NODE} MATCH (b)-[:KNOWS]->(y) RETURN b.name AS b, y.name AS y"),
    ("V2", f"{V_NODE} MATCH (x:Person)-[:KNOWS]->(b) RETURN b.name AS b, x.name AS x"),
    ("V3", f"{V_NODE} MATCH (y:Person) WHERE y = b RETURN y.name AS y"),
    ("V4", "MATCH (a:Person {name: 'Alix'}) CALL { WITH a MATCH (a)-[:KNOWS]->(b) RETURN b, id(b) AS inner } RETURN b.name AS b, id(b) = inner AS same"),
    ("V5", f"{V_EDGE} RETURN type(r) AS t, r.w AS w"),
    ("V6", f"{V_EDGE} MATCH (x)-[r]->(z) RETURN x.name AS x, z.name AS z"),
    ("V7", "CALL { MATCH (c:City) RETURN c } MATCH (p:Person)-[:LIVES_IN]->(c) RETURN p.name AS p, c.name AS c"),
]:
    case(case_id, query)
case("V8", V_NODE, CYPHER)

# W: a GQL CALL subquery sees the outer row's variables, or those its scope clause names
for case_id, query in [
    ("W1", "MATCH (a:Person {name: 'Alix'}) CALL { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS bn } RETURN bn"),
    ("W2", "MATCH ()-[r:KNOWS]->() CALL { MATCH (x)-[r]->(y) RETURN x.name AS xn } RETURN count(*) AS c"),
    ("W3", "MATCH (a:Person) CALL { WITH a WHERE a.age > 30 RETURN a.name AS m } RETURN m"),
    ("W4", "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS bn } RETURN bn"),
    ("W5", "MATCH (a:Person {name: 'Alix'}) CALL () { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS bn } RETURN count(*) AS c"),
    ("W6", "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Paris'}) CALL (a) { MATCH (c:City) RETURN count(c) AS n } RETURN n"),
    ("W7", "MATCH (a:Person) OPTIONAL CALL (a) { MATCH (a)-[:LIVES_IN]->(c) RETURN c.name AS c } RETURN a.name AS a, c"),
]:
    case(case_id, query, GQL)

# X: a Cypher CALL subquery with a variable scope clause
for case_id, query in [
    ("X1", "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS bn } RETURN bn"),
    ("X2", "MATCH (a:Person {name: 'Alix'}) CALL (*) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS bn } RETURN bn"),
    ("X3", "MATCH (a:Person {name: 'Alix'}) CALL () { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS bn } RETURN count(*) AS c"),
    ("X4", "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Paris'}) CALL (a) { MATCH (c:City) RETURN count(c) AS n } RETURN n"),
    ("X5", "MATCH (a:Person) CALL (a) { WITH a WHERE a.age > 30 RETURN a.name AS m } RETURN m"),
]:
    case(case_id, query, CYPHER)

# Y: RETURN * in a CALL subquery returns the variables the subquery binds itself
Y_STAR = "MATCH (a:Person {name: 'Alix'}) CALL { WITH a MATCH (a)-[:KNOWS]->(b) RETURN * }"
for case_id, query, languages in [
    ("Y1", f"{Y_STAR} RETURN a.name AS a, b.name AS b", BOTH),
    ("Y2", f"{Y_STAR} MATCH (x:Person)-[:KNOWS]->(b) RETURN b.name AS b, x.name AS x", BOTH),
    ("Y3", "CALL { MATCH (c:City) RETURN * } RETURN c.name AS c", BOTH),
    ("Y4", "MATCH (a:Person {name: 'Alix'}) CALL { MATCH (a)-[r:KNOWS]->(b) RETURN * } RETURN type(r) AS t, b.name AS b", GQL),
]:
    case(case_id, query, languages)

# Z: a CALL subquery's RETURN under ORDER BY and LIMIT passes on its nodes too
for case_id, query in [
    ("Z1", "MATCH (a:Person {name: 'Alix'}) CALL { MATCH (a)-[:KNOWS]->(b) RETURN b ORDER BY b.name DESC LIMIT 1 } MATCH (b)-[:KNOWS]->(y) RETURN b.name AS b, y.name AS y"),
    ("Z2", "MATCH (a:Person {name: 'Alix'}) CALL { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS n ORDER BY b.age LIMIT 1 } RETURN n"),
]:
    case(case_id, query, GQL)

# AA: one-hop quantifiers bind a list, collected list items stay edges, pattern comprehensions in aggregates
for case_id, query, languages in [
    ("AA1", "MATCH (a:Person {name: 'Alix'})-[rs:KNOWS*1..1]->(b) RETURN b.name AS b, size(rs) AS n", CYPHER),
    ("AA2", "MATCH (a:Person {name: 'Alix'})-[rs:KNOWS]->{1,1}(b) RETURN b.name AS b, size(rs) AS n", GQL),
    ("AA3", "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b) WITH collect(last(relationships(p))) AS es UNWIND es AS e RETURN e.w AS w", BOTH),
    ("AA4", "MATCH (a:Person) RETURN sum(size([(a)-[:KNOWS]->(b) | b])) AS n", CYPHER),
    ("AA5", "MATCH (a:Person) RETURN a.name AS a, size([(a)-[:KNOWS]->(b) | b]) AS n", CYPHER),
    ("AA6", "MATCH (a:Person) RETURN a.name AS a, size(a{.name, knows: [(a)-[:KNOWS]->(b) | b]}.knows) AS n", CYPHER),
]:
    case(case_id, query, languages)

# AB: a CALL subquery returns new names only; a WITH ends the scope of what it leaves out
for case_id, query in [
    ("AB1", "MATCH (a:Person {name: 'Alix'}) CALL { RETURN 1 AS a } RETURN a"),
    ("AB2", "MATCH (a:Person {name: 'Alix'}) CALL (a) { RETURN a } RETURN a.name AS n"),
    ("AB3", "MATCH (a:Person {name: 'Alix'}) WITH 1 AS x RETURN a.name AS n"),
    ("AB4", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH a, count(b) AS k RETURN a.name AS n, k } RETURN n, k"),
    ("AB5", "MATCH (a:Person {name: 'Alix'}) WITH a.name RETURN a.name"),
]:
    case(case_id, query, BOTH)

# AC: subquery bodies: ORDER BY, SKIP, LIMIT and UNION in CALL; OPTIONAL MATCH in and before subqueries
for case_id, query, languages in [
    ("AC1", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS f ORDER BY b.age DESC LIMIT 1 } RETURN a.name AS a, f", BOTH),
    ("AC2", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS f ORDER BY f SKIP 1 } RETURN a.name AS a, f", BOTH),
    ("AC3", "MATCH (a:Person {name: 'Alix'}) CALL (a) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS x UNION MATCH (b)-[:KNOWS]->(a) RETURN b.name AS x } RETURN x", BOTH),
    ("AC4", "MATCH (a:Person {name: 'Alix'}) CALL (a) { RETURN a.name AS x UNION ALL RETURN a.name AS x } RETURN x", BOTH),
    ("AC5", "MATCH (a:Person) CALL { WITH a ORDER BY a.age MATCH (a)-[:KNOWS]->(b) RETURN b.name AS f } RETURN f", CYPHER),
    ("AC6", "RETURN 1 AS x UNION ALL RETURN 1 AS x UNION RETURN 1 AS x", CYPHER),
    ("AC7", "MATCH (a:Person) RETURN a.name AS a, COUNT { MATCH (a)-[:KNOWS]->(b) OPTIONAL MATCH (b)-[:LIVES_IN]->(c) } AS n", BOTH),
    ("AC8", "OPTIONAL MATCH (x:Robot) RETURN x.name AS x", BOTH),
    ("AC9", "MATCH (p:Person) RETURN p.name AS p, VALUE { OPTIONAL MATCH (p)-[:KNOWS]->(f) RETURN count(f) } AS friends", GQL),
    ("AC10", "MATCH (p:Person) RETURN p.name AS p, COUNT { OPTIONAL MATCH (p)-[:KNOWS]->(f) } AS n", BOTH),
    ("AC11", "MATCH (p:Person) WHERE EXISTS { OPTIONAL MATCH (p)-[:LIVES_IN]->(c) } RETURN p.name AS p", BOTH),
    ("AC12", "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b) OPTIONAL MATCH (b)-[:KNOWS]->(c) WHERE c.age > a.age RETURN b.name AS b, c.name AS c", BOTH),
    ("AC13", "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b) OPTIONAL MATCH (b)-[:KNOWS]->(c WHERE c.age > a.age) RETURN b.name AS b, c.name AS c", GQL),
]:
    case(case_id, query, languages)

# AD: a subquery that imports nothing sees no outer variable
for case_id, query, languages in [
    ("AD1", "MATCH (a:Person) CALL () { MATCH (b:Person) WHERE b.age > a.age RETURN count(b) AS c } RETURN a.name AS a, c", BOTH),
    ("AD2", "MATCH (a:Person) CALL { MATCH (b:Person) WHERE b.age > a.age RETURN count(b) AS c } RETURN a.name AS a, c", CYPHER),
    ("AD3", "UNWIND [1] AS a CALL () { MATCH (a) RETURN a AS x } RETURN count(x) AS n", BOTH),
]:
    case(case_id, query, languages)

# AE: a later pattern back to a variable bound before a shortest path, and subqueries that share nothing
for case_id, query, languages in [
    ("AE1", "MATCH p = shortestPath((a:Person {name: 'Gus'})-[:KNOWS*]->(b:Person {name: 'Alix'})) MATCH (b)-[:KNOWS]->(c)-[:KNOWS]->(a) RETURN c.name AS c", CYPHER),
    ("AE2", "MATCH p = ANY SHORTEST (a:Person {name: 'Gus'})-[:KNOWS]->+(b:Person {name: 'Alix'}) MATCH (b)-[:KNOWS]->(c)-[:KNOWS]->(a) RETURN c.name AS c", GQL),
    ("AE3", "MATCH (c:City) RETURN c.name AS c, COUNT { MATCH (x)-[:KNOWS]->(y) } AS n", BOTH),
    ("AE4", "MATCH (c:City) RETURN c.name AS c, EXISTS { MATCH (x)-[:LIVES_IN]->(y:City {name: 'Paris'}) } AS e", BOTH),
]:
    case(case_id, query, languages)

# AF: GQL NEXT passes rows on; a VALUE subquery reads the outer row (read only: the fixtures are shared)
for case_id, query in [
    ("AF1", "MATCH (a:Person {name: 'Alix'}) RETURN a NEXT MATCH (a)-[:KNOWS]->(b) RETURN b.name AS b"),
    ("AF2", "MATCH (a:Person {name: 'Alix'}) RETURN a.age AS x NEXT RETURN x + 1 AS y"),
    ("AF3", "MATCH (p:Person) RETURN p.name AS p, VALUE { MATCH (p)-[:KNOWS]->(f) RETURN f.name ORDER BY f.name LIMIT 1 } AS first"),
    ("AF4", "MATCH (p:Person) WHERE VALUE { MATCH (p)-[:KNOWS]->(f) RETURN f.name ORDER BY f.name LIMIT 1 } IS NOT NULL RETURN p.name AS p"),
]:
    case(case_id, query, GQL)

# AG: node patterns with several labels (the label with the fewest nodes is scanned) and
#     lookups of a labeled node by an indexed property, on skewed labels; AG101 to AG123
#     are AG1 to AG23 with an index on `id`. AG21 to AG23: a label under NOT or OR is
#     no requirement of the node, so the scan keeps the pattern's label
for fixture, offset in [("labels", 0), ("labels_indexed", 100)]:
    for number, query, is_ordered in [
        (1, "MATCH (n:Graph:Repository) RETURN n.id AS id", False),
        (2, "MATCH (n:Repository:Graph) RETURN n.id AS id", False),
        (3, "MATCH (n:Repository:Graph) RETURN n.id AS id ORDER BY id DESC", True),
        (4, "MATCH (n:Graph:Repository) RETURN count(n) AS c", False),
        (5, "MATCH (n:Graph:Repository {id: 'r1'}) RETURN n.id AS id", False),
        (6, "MATCH (n:Repository:Graph {id: 'r1'}) RETURN n.id AS id", False),
        (7, "MATCH (n:Graph:Repository {id: 'g3'}) RETURN n.id AS id", False),
        (8, "MATCH (n:Repository:Graph {id: 'p3'}) RETURN n.id AS id", False),
        (9, "MATCH (n:Graph {id: 'r2'}) RETURN n.id AS id", False),
        (10, "MATCH (n:Graph {id: 'p3'}) RETURN n.id AS id", False),
        (11, "MATCH (n:Graph) WHERE n.id IN ['g19', 'r0', 'p3', 'missing', 'r2'] RETURN n.id AS id", False),
        (12, "MATCH (n:Repository:Graph) WHERE n.id IN ['g19', 'r0', 'p3', 'missing', 'r2'] RETURN n.id AS id", False),
        (13, "MATCH (n:Graph:Repository {n: 2}) RETURN n.id AS id", False),
        (14, "MATCH (n:Graph:Repository) WHERE n.n > 0 RETURN n.id AS id", False),
        (15, "MATCH (n:Graph) WHERE n:Repository RETURN n.id AS id", False),
        (16, "MATCH (n:Repository:Graph)-[r:HAS]->(m) RETURN n.id AS n, r.w AS w, m.id AS m", False),
        (17, "MATCH (a:Graph {id: 'g88'}) MATCH (n:Graph:Repository) RETURN a.id AS a, n.id AS n", False),
        (18, "MATCH (a:Graph {id: 'g88'}) OPTIONAL MATCH (n:Graph:Repository {id: 'p1'}) RETURN a.id AS a, n.id AS n", False),
        (19, "MATCH (n:Tag:Topic) RETURN n.id AS id", False),
        (20, "MATCH (n:Topic:Tag) RETURN n.id AS id", False),
        (21, "MATCH (n:Graph) WHERE NOT n:Repository RETURN count(n) AS c", False),
        (22, "MATCH (n:Graph) WHERE n:Repository OR n.n = 5 RETURN count(n) AS c", False),
        (23, "MATCH (n:Graph) WHERE NOT (n:Repository AND n.n = 1) RETURN count(n) AS c", False),
    ]:
        case(f"AG{offset + number}", query, BOTH, fixture, is_ordered)

# AH: a WHERE conjunct moves only where every variable it reads is bound (#455): a later
#     MATCH's scan, one side of a join, the input of a CALL subquery; a filter on
#     length(p) stays above the expand that binds p. AH9 and AH10 are the shape of #455.
#     AH14 to AH16: a filter on a variable a WITH drops or renames is not copied onto the
#     OPTIONAL MATCH variable of that name. AH17 to AH19: a condition without variables
#     filters the one row a query starts from.
#     AH111 to AH113: a key from the row looked up through the checks of the other labels
#     (AH211 to AH213: the same with an index on `id`)
for case_id, query, languages in [
    ("AH1", "MATCH p = (a:Person)-[:KNOWS]->(b) WHERE length(p) >= 1 RETURN a.name AS a, b.name AS b", BOTH),
    ("AH2", "MATCH p = (a:Person)-[:KNOWS*1..2]->(b) WHERE length(p) = 2 RETURN a.name AS a, b.name AS b", BOTH),
    ("AH3", "MATCH (x:City), (a:Person), p = (a)-[:KNOWS]->(b) WHERE length(p) * 104 >= x.w RETURN x.name AS x, a.name AS a, b.name AS b", BOTH),
    ("AH4", "UNWIND ['Paris'] AS u MATCH (x:City), (a:Person), (a)-[:LIVES_IN]->(c) WHERE c.name = u RETURN x.name AS x, a.name AS a", BOTH),
    ("AH5", "MATCH (a:Person) CALL { WITH a RETURN a.age AS g } WITH * WHERE a.age = g RETURN a.name AS a, g", CYPHER),
    ("AH6", "MATCH (a:Person) MATCH (c:City) CALL { WITH c RETURN c.w AS w } WITH * WHERE a.w + 4 = w RETURN a.name AS a, c.name AS c", CYPHER),
    ("AH7", "MATCH (a:Person) CALL { MATCH (x:City), (y:City) RETURN x, y } WITH * WHERE a.w + 4 = x.w RETURN a.name AS a, x.name AS x, y.name AS y", CYPHER),
    ("AH8", "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIVES_IN]->(c), (x:City) WITH * WHERE c.w = a.w + 4 RETURN a.name AS a, c.name AS c, x.name AS x", CYPHER),
    ("AH9", "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 25 MATCH (x:City), (y:City) WHERE x.w = a.w + 4 AND y.w = b.w + 4 RETURN a.name AS a, b.name AS b, x.name AS x, y.name AS y", CYPHER),
    ("AH10", "MATCH (a:Person)-[:KNOWS]->(b) MATCH (x:City), (y:City) WHERE a.age > 25 AND x.w = a.w + 4 AND y.w = b.w + 4 RETURN a.name AS a, b.name AS b, x.name AS x, y.name AS y", BOTH),
    ("AH14", "MATCH (a:Person) WHERE a.age = 30 WITH a AS x OPTIONAL MATCH (a:City)<-[:LIVES_IN]-(y) RETURN x.name AS x, a.name AS a, y.name AS y", CYPHER),
    ("AH15", "MATCH (a:Person), (c:City) WHERE a.age = 30 WITH c AS a OPTIONAL MATCH (a)<-[:LIVES_IN]-(y) RETURN a.name AS a, y.name AS y", CYPHER),
    ("AH16", "MATCH (a:Person) MATCH (b:Person) WHERE a.age = 30 WITH b OPTIONAL MATCH (a:City)<-[:LIVES_IN]-(y) RETURN b.name AS b, a.name AS a, y.name AS y", BOTH),
    ("AH17", "WITH 1 AS x WHERE 1 = 1 RETURN x", CYPHER),
    ("AH18", "CALL { RETURN 1 AS x } WITH * WHERE 1 = 1 RETURN x", BOTH),
    ("AH19", "OPTIONAL MATCH (a:Person) WITH * WHERE 1 = 1 RETURN a.name AS a", BOTH),
]:
    case(case_id, query, languages)
for fixture, offset in [("labels", 100), ("labels_indexed", 200)]:
    for number, query in [
        (11, "UNWIND ['r0', 'g3', 'p1', 'r2'] AS k MATCH (n:Graph:Repository {id: k}) RETURN k, n.n AS n"),
        (12, "UNWIND ['r0', 'g3', 'p1', 'r2'] AS k MATCH (n:Repository:Graph {id: k}) RETURN k, n.n AS n"),
        (13, "MATCH (a:Repository {id: 'r1'}) CALL (a) { MATCH (n:Graph:Repository {id: a.id}) RETURN n.n AS n } RETURN a.id AS a, n"),
    ]:
        case(f"AH{offset + number}", query, GQL if number == 13 else BOTH, fixture)

# fmt: on
