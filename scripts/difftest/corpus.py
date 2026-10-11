"""The difftest corpus: fixtures and the queries that run on them.

The fixtures are built so that mistakes show. Nodes carry a property `w` of 100 and
up, edges one below 100, so an edge read as a node (or the reverse) gives a value from
the wrong range. `chain` has 5,000 nodes, so sorts, cuts and skips cross row batches.

Each case has an id, the query, the languages it runs in, its fixture, and whether its
rows are ordered (unordered rows compare as a multiset). Queries only read: each fixture
is built once and shared by its cases. The one exception is the `writes` fixture, an empty
database of its own on which the writes of section BR run in order, with reads of what
they wrote. Results are matched by id and
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


def writes(grafeo):
    """An empty database for the cases that write (section BR), which no other case reads."""
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


def labels_compacted(grafeo):
    """`labels` after `compact()`: a columnar table per label set, so `r0` to `r2` and
    `t0` are rows of the tables of their two labels."""
    db = labels(grafeo)
    db.compact()
    return db


def search(grafeo):
    """Files `a.rs` and `b.rs`, and four `Doc` nodes with a 2-dimensional `emb` and a
    `body`: Amsterdam ([1, 0], 'graph database'), Berlin ([0, 1], 'rust compiler'), Paris
    ([0.8, 0.6], 'graph theory') and Prague ([0.3, 0.95], 'query planner'). A cosine vector
    index on `Doc.emb` and a text index on `Doc.body`."""
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:File {name: 'a.rs'}), (:File {name: 'b.rs'})")
    for name, emb, body in [
        ("Amsterdam", "[1.0, 0.0]", "graph database"),
        ("Berlin", "[0.0, 1.0]", "rust compiler"),
        ("Paris", "[0.8, 0.6]", "graph theory"),
        ("Prague", "[0.3, 0.95]", "query planner"),
    ]:
        db.execute(
            f"INSERT (:Doc {{name: '{name}', emb: vector({emb}), body: '{body}'}})"
        )
    db.execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 2 METRIC 'cosine'")
    db.execute("CREATE INDEX doc_body FOR (n:Doc) ON (n.body) USING TEXT")
    return db


def typed(grafeo):
    """Type DDL and the writes after it: `City` with a node type default (`country`
    'NL'), `ROUTE` between cities with an edge type default (`km` 88), Paris to Prague
    without `km`, Berlin to Amsterdam with `km` 3; and a graph type in ISO's brace form
    that declares `Stop` and `LEG`. Every statement runs on 0.5.44 too."""
    db = grafeo.GrafeoDB()
    db.execute("CREATE NODE TYPE City (name STRING, country STRING DEFAULT 'NL')")
    db.execute(
        "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64 DEFAULT 88)"
    )
    db.execute("INSERT (:City {name: 'Paris'})-[:ROUTE]->(:City {name: 'Prague'})")
    db.execute(
        "INSERT (:City {name: 'Berlin'})-[:ROUTE {km: 3}]->(:City {name: 'Amsterdam'})"
    )
    db.execute(
        "CREATE GRAPH TYPE stops "
        "{ (:Stop {name STRING NOT NULL, zone INT64})-[:LEG {minutes INT64}]->(:Stop) }"
    )
    return db


def mixed(grafeo, indexed=False):
    """`Doc` nodes whose `p` is 42 as an integer, a float and three strings ('42', '042',
    '42.0'), two numbers within EPSILON of each other (0.1 + 0.2 and 0.3) and a word,
    each named `n` after its kind, and one `Other` node. With `indexed`, a property index
    on `p`."""
    db = grafeo.GrafeoDB()
    if indexed:
        db.create_property_index("p")
    for name, value in [
        ("int", "42"),
        ("float", "42.0"),
        ("string", "'42'"),
        ("padded", "'042'"),
        ("decimal", "'42.0'"),
        ("sum", "0.1 + 0.2"),
        ("tenths", "0.3"),
        ("word", "'abc'"),
    ]:
        db.execute(f"INSERT (:Doc {{n: '{name}', p: {value}}})")
    db.execute("INSERT (:Other {n: 'other'})")
    return db


def mixed_indexed(grafeo):
    """`mixed` with a property index on `p`."""
    return mixed(grafeo, indexed=True)


FIXTURES = {
    "social": social,
    "chain": chain,
    "empty": empty,
    "writes": writes,
    "labels": labels,
    "labels_indexed": labels_indexed,
    "labels_compacted": labels_compacted,
    "search": search,
    "typed": typed,
    "mixed": mixed,
    "mixed_indexed": mixed_indexed,
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

# J: aggregation over nodes and edges; J4 to J6: a grouped aggregate skips null operands,
#    also the first one of a group
ordered("J1", "MATCH (a:Person)-[r]->(b) RETURN b.name AS n, count(r) AS c ORDER BY c DESC, n")
case("J2", "MATCH (a:Person)-[r]->(b) RETURN r, count(*) AS c")
case("J3", "MATCH (a:Person)-[r]->(b) RETURN collect(r.w) AS ws")
case("J4", "UNWIND [null, 3, 1] AS x RETURN 0 AS g, min(x) AS a, max(x) AS b")
case("J5", "UNWIND [null, 3] AS x RETURN 0 AS g, sample(x) AS s")
case("J6", "UNWIND [null, 19, 3] AS x RETURN 0 AS g, collect(x) AS xs, collect(DISTINCT x) AS ds")
# J7 to J14: values beside a computed group key or aggregate operand keep their kind (they
# were copied through as nodes, so strings, floats, booleans, lists, maps and paths became 0)
case("J7", "MATCH (a:Person) WITH a.name AS name, a.age AS age RETURN age % 2 AS odd, collect(name) AS names")
case("J8", "MATCH (a:Person) WITH a.name AS name, a.age AS age RETURN name, sum(age * 2) AS doubled")
case("J9", "MATCH (a:Person) WITH a.age / 16.0 AS share, a.age > 30 AS senior, [a.name, a.age] AS pair, {age: a.age} AS info RETURN 0 AS g, collect(share) AS shares, collect(senior) AS seniors, collect(pair) AS pairs, collect(info) AS infos")
case("J10", "MATCH (a:Person) WITH a.name AS name, a.age AS age RETURN count(DISTINCT name) AS names, max(name) AS last, sum(age * 2) AS doubled")
case("J11", "UNWIND [2.5, 3.5] AS x RETURN 0 AS g, collect(x) AS xs, sum(x) AS total")
case("J12", "MATCH (a:Person) WITH a.name AS name, a.age AS age WITH age % 2 AS odd, collect(name) AS names RETURN odd, names")
case("J13", "MATCH (a:Person)-[r:KNOWS]->(b) RETURN a.age % 2 AS odd, collect(r) AS rs, collect(b) AS bs, sum(r.w * 2) AS doubled")
case("J14", "MATCH p = (a:Person)-[:KNOWS]->(b) RETURN a.age % 2 AS odd, collect(p) AS ps")

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

# AI: the distinct nodes (#463) a variable-length pattern reaches (DISTINCT, count(DISTINCT), min,
#     max): each node once per source instead of once per walk, round the KNOWS triangle,
#     back along the first edge and across edge types; once over all sources when only the
#     end nodes are read (AI9 and AI11), once per source when the source is read (AI10)
for case_id, query in [
    ("AI1", "MATCH (a:Person {name: 'Alix'})-[*1..2]-(b) RETURN DISTINCT b.name AS b"),
    ("AI2", "MATCH (a:Person {name: 'Alix'})-[:KNOWS*2..3]-(b) RETURN DISTINCT b.name AS b"),
    ("AI3", "MATCH (a:Person)-[:KNOWS*0..2]->(b) RETURN DISTINCT b.name AS b"),
    ("AI4", "MATCH (a:Person)-[:KNOWS*1..3]-(b) RETURN a.name AS a, count(DISTINCT b) AS n"),
    ("AI5", "MATCH (a:Person)-[*1..2]->(b) WHERE b.age > 26 RETURN min(b.age) AS lo, max(b.age) AS hi"),
    ("AI6", "MATCH (a:Person {name: 'Gus'})<-[:KNOWS*1..3]-(b) RETURN DISTINCT b.name AS b"),
    ("AI7", "MATCH (a:Person)-[*1..2]-(c:City) RETURN DISTINCT c.name AS c"),
    ("AI8", "MATCH (a:Person {name: 'Jules'})-[*1..3]-(b) WITH DISTINCT b RETURN b.name AS name, b.w AS w"),
    ("AI9", "MATCH (a:Person)-[:KNOWS*2..3]->(b) RETURN DISTINCT b.name AS b"),
    ("AI10", "MATCH (a:Person)-[:KNOWS*1..3]-(b) RETURN DISTINCT a.name AS a, b.name AS b"),
    ("AI11", "MATCH (a:Person)-[*1..2]-(b) WITH b.name AS name RETURN DISTINCT name"),
]:
    case(case_id, query)

# AJ: a later MATCH joined to an earlier one by equal values runs as a hash join (#455):
#     keys of both kinds of number, numeric strings and NULL (AJ3), more keys, keys
#     computed from properties, conjuncts beside the keys, the checks of more labels
#     (AJ8), and more rows than a batch holds (AJ9)
for case_id, query, languages, fixture in [
    ("AJ1", "MATCH (a:Person) MATCH (c:City) WHERE c.w = a.w + 4 RETURN a.name AS a, c.name AS c", BOTH, "social"),
    ("AJ2", "MATCH (a:Person) MATCH (b:Person) WHERE b.age = a.age RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("AJ3", "UNWIND [{k: 100}, {k: '100'}, {k: 100.0}, {k: '100.0'}, {k: null}, {k: 1e-17}] AS r MATCH (p:Person) WHERE p.w = r.k RETURN r.k AS k, p.name AS p", BOTH, "social"),
    ("AJ4", "MATCH (a:Person) MATCH (b:Person) WHERE b.age = a.age AND b.w = a.w RETURN a.name AS a, b.name AS b", BOTH, "social"),
    ("AJ5", "MATCH (a:Person)-[:LIVES_IN]->(c) MATCH (d:City) WHERE toUpper(d.name) = toUpper(c.name) AND d.w > 100 RETURN a.name AS a, d.name AS d", BOTH, "social"),
    ("AJ6", "MATCH (a:N) MATCH (b:N) WHERE a.i < 50 AND b.m = a.i % 7 AND b.i < 100 RETURN count(*) AS c, sum(b.i) AS s", BOTH, "chain"),
    ("AJ7", "MATCH (a:N) MATCH (b:N) WHERE a.i < 30 AND toString(b.i) = toString(a.m) RETURN a.i AS a, b.i AS b", BOTH, "chain"),
    ("AJ8", "MATCH (r:Repository) MATCH (n:Graph:Repository) WHERE r.n < 3 AND n.n = r.n RETURN r.id AS r, n.id AS n", BOTH, "labels"),
]:
    case(case_id, query, languages, fixture)
ordered("AJ9", "MATCH (a:N) MATCH (b:N) WHERE a.i < 3000 AND b.i = a.i * 2 RETURN a.i AS a, b.i AS b ORDER BY a DESC LIMIT 5", BOTH, "chain")

# AK: a compacted database reads each label of a node with several: labels(n), scans
#     and counts of each label, patterns with several labels, and a traversal from them
for case_id, query in [
    ("AK1", "MATCH (n {id: 'r1'}) RETURN labels(n) AS l"),
    ("AK2", "MATCH (n:Repository) RETURN n.id AS id"),
    ("AK3", "MATCH (n:Graph) RETURN count(n) AS c"),
    ("AK4", "MATCH (n:Graph:Repository) RETURN n.id AS id"),
    ("AK5", "MATCH (n:Repository:Graph)-[r:HAS]->(m) RETURN n.id AS n, r.w AS w, m.id AS m"),
    ("AK6", "MATCH (n:Tag) RETURN n.id AS id"),
    ("AK7", "MATCH (n:Topic:Tag) RETURN n.id AS id"),
    ("AK8", "MATCH (n:Graph) WHERE NOT n:Repository RETURN count(n) AS c"),
    ("AK9", "MATCH (n:Repository) WHERE n.n = 2 RETURN n.id AS id, n.w AS w"),
]:
    case(case_id, query, BOTH, "labels_compacted")

# AM: a part of a later MATCH that reuses a node or edge an earlier clause bound goes on
#     from it, whatever the order of the parts (AM3: the order that already worked), with a
#     WHERE after either MATCH (AM5 in Cypher, AM10 its GQL form inside the pattern), in an
#     OPTIONAL MATCH, in a CALL subquery (AM11: WITH *), with a third part joined on a
#     variable of the second (AM9), and a part that reads an unwound value (AM12)
for case_id, query, languages in [
    ("AM1", "MATCH (a:Person) MATCH (c:City), (a)-[:LIVES_IN]->(c) RETURN a.name AS a, c.name AS c", BOTH),
    ("AM2", "MATCH (a:Person) MATCH (c:City), (c)<-[:LIVES_IN]-(a) RETURN a.name AS a, c.name AS c", BOTH),
    ("AM3", "MATCH (a:Person) MATCH (a)-[:LIVES_IN]->(c), (c:City) RETURN a.name AS a, c.name AS c", BOTH),
    ("AM4", "MATCH (a:Person) MATCH (c:City), (a)-[:LIVES_IN]->(c) WHERE a.age > 26 RETURN a.name AS a, c.name AS c", BOTH),
    ("AM5", "MATCH (a:Person) WHERE a.age < 30 MATCH (c:City), (a)-[:LIVES_IN]->(c) RETURN a.name AS a, c.name AS c", CYPHER),
    ("AM6", "MATCH (a:Person) OPTIONAL MATCH (c:City), (a)-[:LIVES_IN]->(c) RETURN a.name AS a, c.name AS c", BOTH),
    ("AM7", "MATCH (a:Person)-[r:KNOWS]->(b) MATCH (c:Person), (c)<-[r]-(x) RETURN x.name AS x, c.name AS c, r.w AS w", BOTH),
    ("AM8", "MATCH (a:Person) CALL { WITH a MATCH (c:City), (a)-[:LIVES_IN]->(c) RETURN c.name AS c } RETURN a.name AS a, c", BOTH),
    ("AM9", "MATCH (a:Person) MATCH (b:Person), (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(c) RETURN a.name AS a, b.name AS b, c.name AS c", BOTH),
    ("AM10", "MATCH (a:Person WHERE a.age < 30) MATCH (c:City), (a)-[:LIVES_IN]->(c) RETURN a.name AS a, c.name AS c", GQL),
    ("AM11", "MATCH (a:Person) CALL { WITH * MATCH (c:City), (a)-[:LIVES_IN]->(c) RETURN c.name AS c } RETURN a.name AS a, c", BOTH),
    ("AM12", "UNWIND [6, 8] AS w MATCH (a:Person) MATCH (c:City), (a)-[:LIVES_IN {w: w}]->(c) RETURN w, a.name AS a, c.name AS c", BOTH),
]:
    case(case_id, query, languages)

# AN: a vector or text search on a scan that runs for each row of an earlier clause keeps
#     those rows (it checks the condition per row): a second MATCH, a comma-separated part,
#     AND and OR of both kinds, an UNWIND; AN9 and AN10 search without input (unchanged)
for case_id, query in [
    ("AN1", "MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 RETURN f.name AS f, d.name AS d"),
    ("AN2", "MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 RETURN d.name AS d"),
    ("AN3", "MATCH (f:File), (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) >= 0.75 RETURN f.name AS f, d.name AS d"),
    ("AN4", "MATCH (f:File) MATCH (d:Doc) WHERE text_match(d.body, 'graph') RETURN f.name AS f, d.name AS d"),
    ("AN5", "MATCH (f:File), (d:Doc) WHERE text_score(d.body, 'graph') > 0.0 RETURN f.name AS f, d.name AS d"),
    ("AN6", "MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 AND text_match(d.body, 'theory') RETURN f.name AS f, d.name AS d"),
    ("AN7", "MATCH (f:File), (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.9 OR text_match(d.body, 'rust') RETURN f.name AS f, d.name AS d"),
    ("AN8", "UNWIND [3, 19] AS n MATCH (d:Doc) WHERE text_match(d.body, 'graph') RETURN n, d.name AS d"),
    ("AN9", "MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 RETURN d.name AS d"),
    ("AN10", "MATCH (d:Doc) WHERE text_match(d.body, 'graph') RETURN d.name AS d"),
]:
    case(case_id, query, BOTH, "search")

# AO: an aggregate without MATCH aggregates the one row a query starts from (it failed with
#     "Empty plan")
for case_id, query, languages in [
    ("AO1", "RETURN count(*) AS c", BOTH),
    ("AO2", "RETURN 3 AS x, count(*) AS c", BOTH),
    ("AO3", "CALL { RETURN count(*) AS c } RETURN c", BOTH),
    ("AO4", "WITH count(*) AS c WHERE c > 0 RETURN c", CYPHER),
]:
    case(case_id, query, languages)

# AP: a part of a MATCH that shares a node with an earlier part goes on from the rows
#     before it when its property map or inline WHERE reads one of their values: an unwound
#     value (AP1 to AP3, AP5 in GQL), or a node of another part (AP4); it was matched on its
#     own without that value and found nothing. AP6: OPTIONAL MATCH, which already worked
for case_id, query, languages in [
    ("AP1", "UNWIND [6, 8] AS w MATCH (c:City), (c)<-[:LIVES_IN {w: w}]-(a) RETURN w, a.name AS a, c.name AS c", BOTH),
    ("AP2", "UNWIND [6, 8] AS w MATCH (c:City), (a)-[:LIVES_IN {w: w}]->(c) RETURN w, a.name AS a, c.name AS c", BOTH),
    ("AP3", "UNWIND [25, 28] AS g MATCH (c:City), (c)<-[:LIVES_IN]-(a {age: g}) RETURN g, a.name AS a, c.name AS c", BOTH),
    ("AP4", "MATCH (x:Person {name: 'Gus'}), (c:City), (c)<-[:LIVES_IN {years: x.age - 22}]-(a) RETURN x.name AS x, a.name AS a, c.name AS c", BOTH),
    ("AP5", "UNWIND [6, 8] AS w MATCH (c:City), (c)<-[r:LIVES_IN WHERE r.w = w]-(a) RETURN w, a.name AS a, c.name AS c", GQL),
    ("AP6", "UNWIND [6, 7, 9] AS w OPTIONAL MATCH (c:City {name: 'Berlin'}), (c)<-[:LIVES_IN {w: w}]-(a) RETURN w, a.name AS a", BOTH),
]:
    case(case_id, query, languages)

# AR: path semantics. A path variable binds the whole path of a pattern of several edge
#     patterns (AR1 to AR4; it held the last hop only, or failed); DIFFERENT EDGES binds no
#     edge twice and a path mode holds for the whole path (AR5 to AR8; both were ignored);
#     a shortest-path search binds its edge variable, keeps the edge pattern's WHERE in the
#     search, and takes an edge bound before (AR9 to AR13)
for case_id, query, languages in [
    ("AR1", "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN a.name AS a, c.name AS c, length(p) AS len, [n IN nodes(p) | n.name] AS names", BOTH),
    ("AR2", "MATCH p = (a)-[:KNOWS]->()-[:KNOWS]->(c) RETURN a.name AS a, c.name AS c, length(p) AS len", BOTH),
    ("AR3", "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->{1,2}(c) RETURN [n IN nodes(p) | n.name] AS names, length(p) AS len", GQL),
    ("AR4", "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS*1..2]->(c) RETURN [n IN nodes(p) | n.name] AS names, length(p) AS len", CYPHER),
    ("AR5", "MATCH DIFFERENT EDGES (a)-[e1:KNOWS]->(b), (a)-[e2:KNOWS]->(c) RETURN a.name AS a, b.name AS b, c.name AS c", GQL),
    ("AR6", "MATCH TRAIL (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*) AS c", GQL),
    ("AR7", "MATCH ACYCLIC (a)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*) AS c", GQL),
    ("AR8", "MATCH REPEATABLE ELEMENTS TRAIL (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*) AS c", GQL),
    ("AR9", "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[e:KNOWS]->{1,4}(b:Person {name: 'Mia'}) RETURN [x IN e | x.w] AS ws, [n IN nodes(p) | n.name] AS names", GQL),
    ("AR10", "MATCH p = shortestPath((a:Person {name: 'Alix'})-[r:KNOWS*]->(b:Person {name: 'Mia'})) RETURN [x IN r | x.w] AS ws, [n IN nodes(p) | n.name] AS names", CYPHER),
    ("AR11", "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w <> 5]->{1,4}(b:Person {name: 'Mia'}) RETURN length(p) AS len", GQL),
    ("AR12", "MATCH ()-[e:KNOWS]->() MATCH ANY SHORTEST (x)-[e]->(y) RETURN e.w AS w, x.name AS x, y.name AS y", GQL),
    ("AR13", "MATCH p = ALL SHORTEST (a:Person {name: 'Alix'})-[:KNOWS]-+(b:Person {name: 'Vincent'}) RETURN [n IN nodes(p) | n.name] AS names", GQL),
]:
    case(case_id, query, languages)

# AV: `=` finds the same rows with and without a property index, in every plan that runs it
#     (#535): a constant key with and without a label, an IN list, a key from the row, a key
#     after a clause, and an id() from a string. `=` finds a number equal to the strings
#     that parse as it and numbers within EPSILON equal; the lookups of an index found
#     their exact value only, and a constant key on a label scan without an index used a
#     stricter `=` of its own. AV1 to AV8 without an index, AV9 to AV16 with one
for case_id, query, fixture in [
    ("AV1", "MATCH (d:Doc) WHERE d.p = 42 RETURN d.n AS n", "mixed"),
    ("AV2", "MATCH (d:Doc) WHERE d.p = 42.0 RETURN d.n AS n", "mixed"),
    ("AV3", "MATCH (d:Doc) WHERE d.p = '042' RETURN d.n AS n", "mixed"),
    ("AV4", "MATCH (d:Doc) WHERE d.p IN [42, 0.3] RETURN d.n AS n", "mixed"),
    ("AV5", "UNWIND [42.0, 0.3] AS k MATCH (d:Doc) WHERE d.p = k RETURN k, d.n AS n", "mixed"),
    ("AV6", "MATCH (o:Other) MATCH (d:Doc) WHERE d.p = 0.3 RETURN d.n AS n", "mixed"),
    ("AV7", "MATCH (d) WHERE d.p = 42 RETURN d.n AS n", "mixed"),
    ("AV8", "MATCH (a:Doc {n: 'int'}) MATCH (b:Doc) WHERE id(b) = toString(id(a)) RETURN b.n AS n", "mixed"),
    ("AV9", "MATCH (d:Doc) WHERE d.p = 42 RETURN d.n AS n", "mixed_indexed"),
    ("AV10", "MATCH (d:Doc) WHERE d.p = 42.0 RETURN d.n AS n", "mixed_indexed"),
    ("AV11", "MATCH (d:Doc) WHERE d.p = '042' RETURN d.n AS n", "mixed_indexed"),
    ("AV12", "MATCH (d:Doc) WHERE d.p IN [42, 0.3] RETURN d.n AS n", "mixed_indexed"),
    ("AV13", "UNWIND [42.0, 0.3] AS k MATCH (d:Doc) WHERE d.p = k RETURN k, d.n AS n", "mixed_indexed"),
    ("AV14", "MATCH (o:Other) MATCH (d:Doc) WHERE d.p = 0.3 RETURN d.n AS n", "mixed_indexed"),
    ("AV15", "MATCH (d) WHERE d.p = 42 RETURN d.n AS n", "mixed_indexed"),
    ("AV16", "MATCH (a:Doc {n: 'int'}) MATCH (b:Doc) WHERE id(b) = toString(id(a)) RETURN b.n AS n", "mixed_indexed"),
]:
    case(case_id, query, BOTH, fixture)

# AW: a statement nested deeper than the stack allows fails with an error that names the
#     limit (#573): 70 levels of parentheses are beyond the nesting limit of 64 (they ran
#     on a large stack and overflowed a small one); 60 levels still run. A chain of AND,
#     OR, XOR or UNION is a balanced tree, so long ones run: an OR of 100 terms (AW2),
#     a UNION ALL of 100 queries (AW5) and an AND of 100 terms (AW6)
for case_id, query in [
    ("AW1", "RETURN " + "(" * 70 + "3" + ")" * 70 + " AS v"),
    ("AW2", "MATCH (a:Person) WHERE " + " OR ".join(f"a.age = {i}" for i in range(100)) + " RETURN count(*) AS c"),
    ("AW3", "RETURN " + "(" * 60 + "3" + ")" * 60 + " AS v"),
    ("AW4", "MATCH (a:Person) WHERE " + " OR ".join(f"a.age = {i}" for i in range(25, 85)) + " RETURN count(*) AS c"),
    ("AW5", " UNION ALL ".join(f"RETURN {i} AS v" for i in range(100))),
    ("AW6", "MATCH (a:Person) WHERE " + " AND ".join(f"a.age <> {i}" for i in range(100)) + " RETURN count(*) AS c"),
]:
    case(case_id, query)
# AX: a Cypher chain of comparisons is their conjunction, `a < b <= c` is `a < b AND b <= c`
#     (AX1 to AX4; it was null, so a WHERE dropped every row), and GQL rejects one (AX5);
#     datetime({epochMillis: n}) and datetime({epochSeconds: n}) are instants (AX6, AX7) and
#     a temporal value's components read like properties (AX8 to AX10); both were null
for case_id, query, languages in [
    ("AX1", "MATCH (p:Person) WHERE 25 <= p.age < 35 RETURN p.name AS name", CYPHER),
    ("AX2", "RETURN 1 <= 2 < 3 AS a, 3 > 19 >= 1 AS b, 3 <> 19 <> 3 AS c, 1 < null < 3 AS d, 19 < 3 < null AS e", CYPHER),
    ("AX3", "MATCH (p:Person) RETURN sum(CASE WHEN 25 <= p.age < 35 THEN 1 ELSE 0 END) AS inside", CYPHER),
    ("AX4", "MATCH (a)-[k:KNOWS]->(b) WHERE 2012 <= k.since < 2020 RETURN a.name AS a, b.name AS b, k.since AS since", CYPHER),
    ("AX5", "RETURN 1 <= 2 < 3 AS a", GQL),
    ("AX6", "RETURN datetime({epochMillis: 1590433388088}) = datetime('2020-05-25T19:03:08.088Z') AS millis, datetime({epochSeconds: 1590364800}) = datetime('2020-05-25T00:00:00Z') AS seconds", BOTH),
    ("AX7", "MATCH (p:Person) RETURN p.name AS name, month(datetime({epochMillis: p.age * 86400000})) AS m, day(datetime({epochMillis: p.age * 86400000})) AS d", BOTH),
    ("AX8", "WITH date('2020-05-25') AS d RETURN d.year AS y, d.quarter AS q, d.month AS m, d.week AS w, d.day AS dd, d.dayOfWeek AS dow, d.ordinalDay AS od", CYPHER),
    ("AX9", "MATCH (p:Person) WITH p, datetime({epochMillis: p.age * 86400000}) AS b WHERE b.month = 1 AND b.day > 28 RETURN p.name AS name, b.day AS day, b.epochSeconds AS s", CYPHER),
    ("AX10", "WITH duration({years: 1, months: 3, days: 19, hours: 3, minutes: 8}) AS d RETURN d.years AS y, d.months AS m, d.monthsOfYear AS moy, d.weeks AS w, d.minutes AS mi, d.minutesOfHour AS moh", CYPHER),
]:
    case(case_id, query, languages)

# AY: DISTINCT in the statistical aggregates (stDev, stDevP, variance, the percentiles and
#     the binary set functions) drops the copies of a value or a pair before the aggregate
#     sees them; they counted every copy (AY1 to AY3, AY7). In GQL an aggregate in HAVING is
#     computed per group, also one the RETURN list does not compute (AY4, AY5; there were no
#     rows), and GROUP BY without an aggregate gives one row per group (AY6; it was ignored);
#     HAVING reads a grouping key by its text or its alias (AY8; there were no rows)
for case_id, query, languages in [
    ("AY1", "MATCH (a:Person)-[:KNOWS]->(b) RETURN stDev(DISTINCT a.age) AS s, stDevP(DISTINCT a.age) AS p, stDev(a.age) AS plain", BOTH),
    ("AY2", "MATCH (a:Person)-[:KNOWS]->(b) RETURN percentileDisc(DISTINCT a.age, 0.25) AS d, percentileCont(DISTINCT a.age, 0.5) AS c", BOTH),
    ("AY3", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name IN ['Alix', 'Gus'] AS front, variance(DISTINCT a.age) AS v", BOTH),
    ("AY4", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a, count(*) AS c GROUP BY a.name HAVING count(*) > 1", GQL),
    ("AY5", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a GROUP BY a.name HAVING min(b.age) < 30", GQL),
    ("AY6", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a GROUP BY a.name", GQL),
    ("AY7", "MATCH (a:Person)-[:KNOWS]->(b) RETURN regr_count(DISTINCT a.age, a.age) AS n, covar_pop(DISTINCT a.age, a.age) AS c", GQL),
    ("AY8", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, count(*) AS c GROUP BY a.name HAVING who <> 'Alix' AND a.name <> 'Gus'", GQL),
]:
    case(case_id, query, languages)

# AZ: values of earlier clauses inside an OPTIONAL MATCH and a comma MATCH. An OPTIONAL
#     MATCH keeps every row: a property map or WHERE that reads an imported value in a CALL
#     subquery (AZ2 to AZ4) or the start of a shortest path (AZ12, AZ13) matched nothing or
#     dropped the row; a WHERE on a list of a WITH with a name the WITH dropped (AZ5, LDBC
#     IC5; AZ6 its GQL form inside the pattern, AZ17 a condition on that name alone), on a
#     node both sides name (AZ7), only on earlier values (AZ8) or on none (AZ9) filtered
#     the rows. GQL's FILTER filters every row (AZ10, AZ11), and so does a WHERE after a
#     questioned edge (AZ14). A later comma part reads a value of the WITH before it (AZ15,
#     AZ16, LDBC IC6); AZ1 already worked
for case_id, query, languages in [
    ("AZ1", "UNWIND [25, 40, 88] AS x OPTIONAL MATCH (p:Person {age: x}) RETURN x, p.name AS p", BOTH),
    ("AZ2", "UNWIND [25, 88] AS x CALL (x) { OPTIONAL MATCH (p:Person {age: x}) RETURN p.name AS p } RETURN x, p", BOTH),
    ("AZ3", "UNWIND [25, 88] AS x CALL { WITH * OPTIONAL MATCH (p:Person WHERE p.age = x) RETURN p.name AS p } RETURN x, p", GQL),
    ("AZ4", "UNWIND [25, 88] AS x CALL { WITH x OPTIONAL MATCH (p:Person) WHERE p.age = x RETURN p.name AS p } RETURN x, p", CYPHER),
    ("AZ5", "MATCH (c:City) MATCH (friend:Person {name: 'Gus'}) WITH c, collect(friend) AS friends OPTIONAL MATCH (friend)-[:LIVES_IN]->(c) WHERE friend IN friends RETURN c.name AS c, count(friend) AS n", CYPHER),
    ("AZ6", "MATCH (c:City) MATCH (friend:Person {name: 'Gus'}) WITH c, collect(friend) AS friends OPTIONAL MATCH (friend WHERE friend IN friends)-[:LIVES_IN]->(c) RETURN c.name AS c, count(friend) AS n", GQL),
    ("AZ7", "MATCH (c:City) OPTIONAL MATCH (p:Person)-[:LIVES_IN]->(c) WHERE c.name = 'Berlin' RETURN c.name AS c, p.name AS p", BOTH),
    ("AZ8", "MATCH (c:City) OPTIONAL MATCH (p:Person {name: 'Mia'}) WHERE c.name = 'Paris' RETURN c.name AS c, p.name AS p", BOTH),
    ("AZ9", "MATCH (c:City) OPTIONAL MATCH (p:Person)-[:LIVES_IN]->(c) WHERE 3 = 19 RETURN c.name AS c, p.name AS p", BOTH),
    ("AZ10", "MATCH (c:City) OPTIONAL MATCH (p:Person)-[:LIVES_IN]->(c) FILTER c.name = 'Berlin' RETURN c.name AS c, p.name AS p", GQL),
    ("AZ11", "MATCH (c:City) OPTIONAL MATCH (p:Person)-[:LIVES_IN]->(c) FILTER p.name = 'Gus' RETURN c.name AS c, p.name AS p", GQL),
    ("AZ12", "UNWIND ['Alix', 'Jules'] AS n OPTIONAL MATCH p = shortestPath((a:Person {name: n})-[:KNOWS*]->(b:Person {name: 'Vincent'})) RETURN n, length(p) AS hops", CYPHER),
    ("AZ13", "UNWIND ['Alix', 'Jules'] AS n OPTIONAL MATCH p = ANY SHORTEST (a:Person {name: n})-[:KNOWS]->+(b:Person {name: 'Vincent'}) RETURN n, length(p) AS hops", GQL),
    ("AZ14", "MATCH (a:Person)-[:KNOWS]->?(b) WHERE b.name = 'Gus' RETURN a.name AS a, b.name AS b", GQL),
    ("AZ15", "MATCH (v:Person {name: 'Vincent'}) WITH v.age AS age MATCH (a:Person {name: 'Gus'}), (a)-[:KNOWS]->(b:Person {age: age}) RETURN b.name AS b", BOTH),
    ("AZ16", "MATCH (:Person {name: 'Gus'})-[r:KNOWS]->() WITH r.since AS y MATCH (a:Person), (a)-[:KNOWS {since: y}]->(b) RETURN a.name AS a, b.name AS b", BOTH),
    ("AZ17", "MATCH (c:City), (p:Person) WITH c, count(p) AS people OPTIONAL MATCH (p)-[:LIVES_IN]->(c) WHERE p.age = 25 RETURN c.name AS c, people, p.name AS p", CYPHER),
]:
    case(case_id, query, languages)

# BB: what type DDL declares (fixture `typed`). An edge type's default fills a property
#     an insert left out, as a node type's does (BB1; the edge had no `km`); a graph type
#     in the brace form declares its element types with their properties (BB2, BB3; it
#     declared their names only)
for case_id, query, languages in [
    ("BB1", "MATCH (a:City)-[r:ROUTE]->(b:City) RETURN a.name AS a, r.km AS km, b.country AS country", BOTH),
    ("BB2", "SHOW NODE TYPES", GQL),
    ("BB3", "SHOW EDGE TYPES", GQL),
]:
    case(case_id, query, languages, fixture="typed")

# BC: path search prefixes. ANY keeps one path per pair of endpoints and input row (BC1 to
#     BC4, BC10; it kept one row in total, and `p = ANY` and ANY k were ignored), with the
#     edge pattern's WHERE checked before it selects (BC4); `p = TRAIL (...)` is the path
#     mode of one pattern (BC5; a syntax error); SHORTEST k and SHORTEST k GROUPS keep k
#     paths or groups per pair (BC6, BC7; one path); a search prefix's path mode restricts
#     the paths it selects among (BC8; ignored), BC9 the WALK control
for case_id, query, languages in [
    ("BC1", "MATCH ANY (a:Person)-[:KNOWS]->{1,3}(b) RETURN a.name AS a, b.name AS b", GQL),
    ("BC2", "MATCH p = ANY 2 (a:Person {name: 'Alix'})-[:KNOWS]-{1,3}(b) RETURN b.name AS b, count(*) AS n", GQL),
    ("BC3", "UNWIND [1, 1] AS x MATCH ANY (a:Person {name: 'Alix'})-[:KNOWS]->{1,2}(b) RETURN x, b.name AS b", GQL),
    ("BC4", "MATCH p = ANY (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w > 1]->{1,3}(b) RETURN b.name AS b, [x IN e | x.w] AS ws", GQL),
    ("BC5", "MATCH p = TRAIL (a:Person {name: 'Alix'})-[:KNOWS]-{1,3}(b:Person {name: 'Vincent'}) RETURN length(p) AS len", GQL),
    ("BC6", "MATCH p = SHORTEST 2 (a:Person {name: 'Alix'})-[:KNOWS]-{1,4}(b:Person) RETURN b.name AS b, length(p) AS len", GQL),
    ("BC7", "MATCH p = SHORTEST 2 GROUPS (a:Person {name: 'Alix'})-[:KNOWS]-{1,4}(b:Person) RETURN b.name AS b, length(p) AS len", GQL),
    ("BC8", "MATCH p = ANY SHORTEST TRAIL (a:Person {name: 'Alix'})-[:KNOWS]-{3,}(b:Person {name: 'Gus'}) RETURN length(p) AS len", GQL),
    ("BC9", "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[:KNOWS]-{3,}(b:Person {name: 'Gus'}) RETURN length(p) AS len", GQL),
    ("BC10", "MATCH ANY (a:Person)-[e:KNOWS]->(b) RETURN a.name AS a, b.name AS b", GQL),
]:
    case(case_id, query, languages)

# BD: GQL statements in any order (#483). An ORDER BY and LIMIT (or OFFSET) before the
#     result statement cut the rows the statements after them read (BD1 to BD4, BD11,
#     BD12 on `chain`); a WHERE or FILTER between MATCH statements filters the rows so far
#     (BD5 to BD8); a statement may start with LET or FILTER, and a WITH may follow a FOR
#     (BD9, BD10). All were syntax errors in GQL; BD2 and BD3 also run in Cypher
for case_id, query, languages, fixture in [
    ("BD1", "MATCH (p:Person) ORDER BY p.age DESC LIMIT 2 MATCH (p)-[:KNOWS]->(f) RETURN p.name AS p, f.name AS f", GQL, "social"),
    ("BD2", "MATCH (p:Person) WITH p ORDER BY p.age DESC LIMIT 2 MATCH (p)-[:KNOWS]->(f) RETURN p.name AS p, f.name AS f", BOTH, "social"),
    ("BD3", "MATCH (p:Person) WITH p LIMIT 1 RETURN count(*) AS c", BOTH, "social"),
    ("BD4", "MATCH (p:Person) ORDER BY p.age OFFSET 1 LIMIT 2 RETURN p.name AS name", GQL, "social"),
    ("BD5", "MATCH (a:Person) WHERE a.age > 29 MATCH (a)-[:LIVES_IN]->(c) RETURN a.name AS a, c.name AS c", GQL, "social"),
    ("BD6", "MATCH (a:Person) FILTER a.age > 29 OPTIONAL MATCH (a)-[:LIVES_IN]->(c) RETURN a.name AS a, c.name AS c", GQL, "social"),
    ("BD7", "MATCH (a:Person) WHERE a.age < 30 CALL (a) { MATCH (a)-[:KNOWS]->(b) RETURN b.name AS b } RETURN a.name AS a, b", GQL, "social"),
    ("BD8", "MATCH (s:Person) WHERE s.age IN [25, 28] LET x = s.name RETURN x", GQL, "social"),
    ("BD9", "LET x = 19 FILTER x > 3 RETURN x + 3 AS y", GQL, "social"),
    ("BD10", "FOR x IN [3, 19, 88] WITH x WHERE x > 3 RETURN x", GQL, "social"),
    ("BD11", "MATCH (n:N) ORDER BY n.i DESC LIMIT 3 MATCH (n)<-[:NEXT]-(m) RETURN n.i AS n, m.i AS m", GQL, "chain"),
    ("BD12", "MATCH (n:N) WHERE n.m = 3 ORDER BY n.i SKIP 700 LIMIT 2 RETURN n.i AS i", GQL, "chain"),
]:
    case(case_id, query, languages, fixture)

# BE: nodes and edges keep their kind inside list and map values (BE1 to BE5: a list or
#     map literal held their IDs, so `{msg: m}.msg.name` was null and RETURN gave numbers;
#     BE3 is the shape of LDBC SNB IC7), startNode and endNode return nodes (BE6; their
#     properties were an error), a returned path holds its nodes and edges (BE7, BE8; it
#     held IDs, and a grouped path became the text `Path(2 nodes, 1 edges)`), and size() of
#     a string counts characters (BE9; it counted UTF-8 bytes)
for case_id, query, languages in [
    ("BE1", "MATCH (a:Person {name: 'Jules'})-[r:KNOWS]->(b) WITH [a, r, b, 3] AS l RETURN l, l[0].w AS aw, l[1].w AS rw, l[2].name AS b", BOTH),
    ("BE2", "MATCH (m:Person {name: 'Mia'})<-[r:KNOWS]-() WITH {msg: m, e: r, t: 19} AS x RETURN x, x.msg.name AS n, x.msg.w AS mw, x.e.w AS ew", BOTH),
    ("BE3", "MATCH (a:Person)-[k:KNOWS]->(b) WITH a, b, k.w AS w ORDER BY w DESC WITH a, head(collect({msg: b, w: w})) AS latest RETURN a.name AS a, latest.msg.name AS msg, latest.msg.w AS mw, latest.w AS w", CYPHER),
    ("BE4", "MATCH (a:Person)-[:KNOWS]->(b) WITH {p: a} AS x, count(b) AS c RETURN x, x.p.w AS w, c", BOTH),
    ("BE5", "MATCH (a:Person)-[r:LIVES_IN]->(c) WITH collect({p: a, e: r}) AS xs UNWIND xs AS x RETURN x, x.p.name AS p, x.e.w AS w", BOTH),
    ("BE6", "MATCH ()-[r:LIVES_IN]->() RETURN startNode(r) AS s, startNode(r).w AS sw, endNode(r).name AS e, labels(endNode(r)) AS l", BOTH),
    ("BE7", "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->()-[:LIVES_IN]->() RETURN p", BOTH),
    ("BE8", "MATCH p = (:Person {name: 'Jules'})-[:KNOWS]->() RETURN p, count(*) AS c", BOTH),
    ("BE9", "MATCH (c:City) RETURN c.name AS name, size(c.name + 'ň') AS chars, size('🌷') AS tulip", BOTH),
]:
    case(case_id, query, languages)
# BF: a filter reaches the scan or seek of its own variable. A target the WHERE pins by
#     ID (BF1 to BF3, BF6, BF7; in GQL the WHERE of the node pattern, BF11 to BF17, as
#     GQL does not read a WHERE after a MATCH that follows a WITH) or by an indexed key
#     (BF208, BF209) starts its expand, which follows the edges back to the source;
#     RETURN * keeps the columns in their order (BF7, BF17). A WHERE after an OPTIONAL
#     MATCH and a WITH filters the rows: its conjunct on the earlier clauses filters them
#     first, the one on the optional side stays above the join (BF4, BF5). Endpoints
#     pinned by row keys are each sought (BF10). These were slower before, the rows are
#     the same
for case_id, query, languages in [
    ("BF1", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)-[r]->(tgt) WHERE id(tgt) = gid RETURN src.name AS s, r.w AS w", CYPHER),
    ("BF2", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)<-[r]-(tgt) WHERE id(tgt) = gid RETURN src.name AS s, type(r) AS t", CYPHER),
    ("BF3", "MATCH (p:Person) WHERE p.name IN ['Gus', 'Mia'] WITH collect(id(p)) AS ids MATCH (src)-[r]-(tgt) WHERE id(tgt) IN ids RETURN src.name AS s, tgt.name AS t, r.w AS w", CYPHER),
    ("BF4", "MATCH (p:Person) OPTIONAL MATCH (p)-[:LIVES_IN]->(c) WITH p, c WHERE p.age > 26 AND c IS NULL RETURN p.name AS p", BOTH),
    ("BF5", "MATCH (p:Person) OPTIONAL MATCH (p)-[:LIVES_IN]->(c) WITH p, c WHERE c.name = 'Amsterdam' AND p.age > 20 RETURN p.name AS p, c.name AS c", BOTH),
    ("BF6", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)-[:KNOWS]->(tgt)-[:LIVES_IN]->(c) WHERE id(tgt) = gid RETURN src.name AS s, c.name AS c", CYPHER),
    ("BF7", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) = gid RETURN *", CYPHER),
    ("BF10", "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) UNWIND [{src: id(a), dst: id(b)}, {src: id(b), dst: id(a)}] AS row MATCH (s), (d) WHERE id(s) = row.src AND id(d) = row.dst RETURN s.name AS s, d.name AS d", BOTH),
    ("BF11", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)-[r]->(tgt WHERE id(tgt) = gid) RETURN src.name AS s, r.w AS w", GQL),
    ("BF12", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)<-[r]-(tgt WHERE id(tgt) = gid) RETURN src.name AS s, type(r) AS t", GQL),
    ("BF13", "MATCH (p:Person) WHERE p.name IN ['Gus', 'Mia'] WITH collect(id(p)) AS ids MATCH (src)-[r]-(tgt WHERE id(tgt) IN ids) RETURN src.name AS s, tgt.name AS t, r.w AS w", GQL),
    ("BF16", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)-[:KNOWS]->(tgt WHERE id(tgt) = gid)-[:LIVES_IN]->(c) RETURN src.name AS s, c.name AS c", GQL),
    ("BF17", "MATCH (g:Person {name: 'Gus'}) WITH id(g) AS gid MATCH (src)-[r:KNOWS]->(tgt WHERE id(tgt) = gid) RETURN *", GQL),
]:
    case(case_id, query, languages)
for fixture, offset in [("labels", 100), ("labels_indexed", 200)]:
    for number, query in [
        (8, "MATCH (src)-[h:HAS]->(g) WHERE g.id IN ['g0', 'g2', 'g7'] RETURN src.id AS s, g.id AS g, h.w AS w"),
        (9, "UNWIND ['g1', 'r1'] AS k MATCH (src)-[h]-(g) WHERE g.id = k RETURN k, src.id AS s"),
    ]:
        case(f"BF{offset + number}", query, BOTH, fixture)
# BG: Cypher translator fixes. exists() of a pattern is true when the pattern has a match
#     for the row (BG1, BG2; it was true for every row); a pattern comprehension reads the
#     row's variables at either end of its pattern and in its filters, and starts from a
#     node of its own when the row binds none (BG3, BG4 failed with Undefined variable
#     imported into CALL; BG5 matched any node at the bound end); the ORDER BY of a WITH
#     that neither aggregates nor is DISTINCT reads its input's variables (BG6, BG7 failed
#     with Undefined variable); a variable named like an anonymous one is the user's (BG9;
#     it matched only self-loops); `*` takes more items after it (BG10, BG11; a syntax
#     error); an aggregate in the ORDER BY of a projection that does not aggregate is an
#     error (BG12 failed with Empty plan, BG13 sorted as if it were not there); after an
#     aggregating RETURN, a key that repeats a returned expression sorts by its column
#     (BG14 sorted as if the key were not there, BG15 failed with an internal error)
for case_id, query, languages in [
    ("BG1", "MATCH (p:Person) RETURN p.name AS name, exists((p)-[:LIVES_IN]->()) AS lives", CYPHER),
    ("BG2", "MATCH (p:Person) WHERE NOT exists((p)-[:KNOWS]->(:Person {name: 'Gus'})) RETURN p.name AS name", CYPHER),
    ("BG3", "RETURN [(a:Person)-[:LIVES_IN]->(c:City {name: 'Paris'}) | a.name] AS names", CYPHER),
    ("BG4", "UNWIND [25, 40] AS x RETURN x, [(p:Person {age: x}) | p.name] AS names", CYPHER),
    ("BG5", "MATCH (a:Person), (b:Person) WHERE a.name IN ['Alix', 'Jules'] AND b.name IN ['Gus', 'Mia'] RETURN a.name AS a, b.name AS b, size([(a)-[:KNOWS]->(b) | 1]) AS links", CYPHER),
    ("BG9", "MATCH (_anon_0:Person)-[:KNOWS]->() RETURN _anon_0.name AS name", CYPHER),
    ("BG10", "MATCH (a:Person)-[r:KNOWS]->(b) WITH *, r.since AS since RETURN a.name AS a, since", CYPHER),
    ("BG11", "UNWIND [3, 19] AS x RETURN *, x * 2 AS y", CYPHER),
    ("BG12", "RETURN 3 AS x ORDER BY count(*)", CYPHER),
    ("BG13", "UNWIND [3, 19] AS x RETURN x ORDER BY count(*)", CYPHER),
]:
    case(case_id, query, languages)
ordered("BG6", "MATCH (a:Person) WITH a.name AS name ORDER BY a.age RETURN name", CYPHER)
ordered("BG7", "MATCH (a:Person) WITH a.name AS name ORDER BY a.age DESC SKIP 1 LIMIT 2 RETURN name", CYPHER)
ordered("BG14", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, max(b.age) AS oldest ORDER BY max(b.age), who", CYPHER)
ordered("BG15", "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, count(*) AS c ORDER BY a.name DESC", CYPHER)

# BH: grouping keys and DISTINCT use equivalence, min and max the order of ORDER BY, and
#     UNWIND keeps its list variable. A float key kept its bits as an integer after a
#     WITH (BH1, 1.5 became 4609434218613702656), 3 and 3.0 were two groups and two
#     distinct values (BH3, BH6, BH11), DISTINCT took paths of one length for one path
#     (BH4, BH5: two paths gave 1), min and max of mixed values depended on the input order
#     and did not compare lists (BH7, BH8), and the list variable read null after UNWIND
#     (BH9, BH10). Cypher's two-argument aggregates read their second argument (BH13, null
#     and 0) and listagg and group_concat their separator (BH14, a space)
for case_id, query, languages in [
    ("BH1", "UNWIND [1.5, 1.5, 2.5] AS x WITH x, count(*) AS c RETURN x, c", BOTH),
    ("BH2", "UNWIND [1.5, 1.5, 2.5] AS x RETURN x, count(*) AS c", BOTH),
    ("BH3", "UNWIND [3, 3.0, 19] AS x WITH x, count(*) AS c RETURN x, c", BOTH),
    ("BH4", "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(DISTINCT p) AS d, count(p) AS n", BOTH),
    ("BH5", "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->()-[:KNOWS]->(c) WITH DISTINCT p RETURN count(*) AS d", BOTH),
    ("BH6", "UNWIND [3, 3.0, 19] AS x RETURN DISTINCT x", BOTH),
    ("BH7", "UNWIND ['a', 3, 1] AS x RETURN min(x) AS lo, max(x) AS hi", BOTH),
    ("BH8", "UNWIND [[1, 2], [1], [0, 88]] AS x RETURN min(x) AS lo, max(x) AS hi", BOTH),
    ("BH9", "MATCH (n:Person) WITH collect(n.name) AS names UNWIND names AS x RETURN size(names) AS s, x", BOTH),
    ("BH10", "WITH [3, 19] AS l UNWIND l AS x RETURN l, x", CYPHER),
    ("BH11", "RETURN 3 AS x UNION RETURN 3.0 AS x", BOTH),
    ("BH12", "LET l = [3, 19] FOR x IN l RETURN l, x", GQL),
    ("BH13", "MATCH (p:Person) RETURN covar_pop(p.age, p.age) AS c, regr_count(p.age, p.age) AS n", BOTH),
    ("BH14", "UNWIND ['Alix', 'Gus'] AS n RETURN listagg(n) AS a, group_concat(n, '|') AS g, group_concat(n) AS s", BOTH),
]:
    case(case_id, query, languages)

# BI: quantified paths. The element pattern WHERE of a quantified edge holds for each edge
#     (BI1, BI2 with an earlier variable; there were no rows); an aggregate over a group
#     variable is computed per path (BI3 to BI5: every internal column and a sum of 0.0,
#     or an undefined variable); a SIMPLE path ends once back at its start (BI6; it went
#     on); `length` reads a path a WITH passed on or an UNWIND gave, and a string (BI7 to
#     BI9; an undefined `_path_length_` column), and an unaliased `length(p)` is named so
#     (BI10; it was `_path_length_p`)
for case_id, query, languages in [
    ("BI1", "MATCH (a:Person)-[e:KNOWS WHERE e.w > 1]->{1,3}(b) RETURN a.name AS a, b.name AS b, [x IN e | x.w] AS ws", GQL),
    ("BI2", "UNWIND [1, 3] AS lim MATCH (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w > lim]->{1,3}(b) RETURN lim, b.name AS b, [x IN e | x.w] AS ws", GQL),
    ("BI3", "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN b.name AS b, sum(e.w) AS total, max(e.w) AS high", GQL),
    ("BI4", "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN sum(e.w)", GQL),
    ("BI5", "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN sum(e.w) > 4 AS long, count(*) AS paths", GQL),
    ("BI6", "MATCH SIMPLE (a:Person {name: 'Alix'})-[:KNOWS]-{1,4}(b) RETURN b.name AS b", GQL),
    ("BI7", "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..3]->(b) WITH b, p WHERE length(p) >= 2 RETURN b.name AS b, length(p) AS len", BOTH),
    ("BI8", "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b) WITH collect(p) AS ps UNWIND ps AS u RETURN length(u) AS len, [n IN nodes(u) | n.name] AS names", BOTH),
    ("BI9", "UNWIND ['Amsterdam', 'Paris'] AS s RETURN s, length(s) AS len", BOTH),
    ("BI10", "MATCH p = (a:Person {name: 'Jules'})-[:KNOWS]->(b) RETURN length(p)", BOTH),
]:
    case(case_id, query, languages)
# BJ: Cypher binds a relationship once per MATCH (openCypher 9, relationship uniqueness),
#     as GQL's MATCH DIFFERENT EDGES does: two patterns never take one relationship (BJ1,
#     BJ2; they came back over it), a variable-length relationship follows trails (BJ3
#     went round the triangle for a hundred hops, BJ12 came back to Mia over the edge it
#     left by), as do OPTIONAL MATCH, a pattern comprehension and EXISTS (BJ6 to BJ8).
#     An unbounded pattern in Cypher or a GQL TRAIL ends on its own instead of stopping
#     after a hundred hops (BJ4, BJ5; 0). BJ11 is the control: a directed DISTINCT reaches
#     what it did. GQL's names for anonymous elements are the statement's own (BJ9, the
#     GQL form of BG9), and an undefined variable named like one is undefined (BJ10; an
#     internal error)
for case_id, query, languages, fixture in [
    ("BJ1", "MATCH (a:Person)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN a.name AS a, b.name AS b, c.name AS c", CYPHER, "social"),
    ("BJ2", "MATCH (a:Person)-[:KNOWS]->(b)<-[:KNOWS]-(c) RETURN a.name AS a, b.name AS b, c.name AS c", CYPHER, "social"),
    ("BJ3", "MATCH (g:Person {name: 'Gus'})-[:KNOWS*]->(c) RETURN c.name AS c", CYPHER, "social"),
    ("BJ4", "MATCH (a:N {i: 0})-[:NEXT*]->(b:N {i: 150}) RETURN count(*) AS c", CYPHER, "chain"),
    ("BJ5", "MATCH TRAIL (a:N {i: 0})-[:NEXT]->{1,}(b:N {i: 150}) RETURN count(*) AS c", GQL, "chain"),
    ("BJ6", "MATCH (m:Person {name: 'Mia'}) OPTIONAL MATCH (m)-[:KNOWS]-(x)-[:KNOWS]-(y) RETURN m.name AS m, x.name AS x, y.name AS y", CYPHER, "social"),
    ("BJ7", "MATCH (m:Person {name: 'Mia'}) RETURN [(m)-[:KNOWS]-(x)-[:KNOWS]-(y) | y.name] AS ys", CYPHER, "social"),
    ("BJ8", "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:KNOWS]-(x)-[:KNOWS]-(p) } RETURN p.name AS p", CYPHER, "social"),
    ("BJ9", "MATCH (_anon_0:Person)-[:KNOWS]->() RETURN _anon_0.name AS name", GQL, "social"),
    ("BJ10", "MATCH (p:Person) RETURN _anon_5", BOTH, "social"),
    ("BJ11", "MATCH (a:Person {name: 'Alix'})-[:KNOWS*]->(b) RETURN DISTINCT b.name AS b", CYPHER, "social"),
    ("BJ12", "MATCH (a:Person {name: 'Mia'})-[:KNOWS*1..2]-(b) RETURN DISTINCT b.name AS b", CYPHER, "social"),
]:
    case(case_id, query, languages, fixture)
# BK: procedure arguments and DFS. A constant expression in a CALL argument runs like its
#     value (BK1, BK9; it ran with the default, or failed as "required"), an integer runs
#     where a float is expected (BK2; damping 1 ran as 0.85), an argument that cannot be
#     evaluated, reads a row or has the wrong type is an error (BK5 to BK7; it ran with the
#     default), and a computed list element is kept (BK8: three elements, so the 2-dimensional
#     index refuses them; 0.0 + 0.0 was dropped and the search ran on two). DFS reports the
#     depth in the DFS tree and its discovery and finish order (BK3, BK4; depth was the
#     finish order)
for case_id, query, languages in [
    ("BK5", "CALL grafeo.pagerank(1 / 0) YIELD score RETURN score", BOTH),
    ("BK6", "MATCH (p:Person {name: 'Alix'}) WITH id(p) AS s CALL grafeo.bfs(s) YIELD node_id RETURN node_id", CYPHER),
    ("BK7", "CALL grafeo.pagerank('high') YIELD score RETURN score", BOTH),
]:
    case(case_id, query, languages)
ordered("BK1", "CALL grafeo.pagerank(0.25 + 0.25, 3 - 2, 0.0001) YIELD node_id, score RETURN node_id, score ORDER BY node_id")
ordered("BK2", "CALL grafeo.pagerank(1, 3) YIELD node_id, score RETURN node_id, score ORDER BY node_id")
ordered("BK3", "CALL grafeo.dfs(0) YIELD node_id, depth RETURN node_id, depth ORDER BY node_id")
ordered("BK4", "CALL grafeo.dfs(0)")
case("BK8", "CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0, 0.0 + 0.0], 2) YIELD node_id RETURN node_id", BOTH, "search")
ordered("BK9", "CALL grafeo.bfs(0 + 0) YIELD node_id, depth RETURN node_id, depth ORDER BY node_id")
# BM: a list literal has one item per expression and a map literal one entry per key. A
#     property of a null entity or a property the entity does not have was left out, so
#     the list was shorter (BM1 `[]`, BM2 `[35]`, BM5 read the node at the wrong position),
#     `[p.w] = []` matched (BM3), the map lost the key (BM4 `{age: 35}`) and IN was false
#     instead of unknown (BM6). A questioned edge without a match is still left out of
#     its path (BM7, unchanged)
for case_id, query, languages in [
    ("BM1", "MATCH (p:Person {name: 'Vincent'}) OPTIONAL MATCH (p)-[r:LIVES_IN]->(c:City) RETURN [c.name, r.years] AS l, size([c.name, r.years]) AS s", BOTH),
    ("BM2", "MATCH (p:Person {name: 'Jules'}) RETURN [p.w, p.age] AS l, [p.w, p.age][1] AS second", BOTH),
    ("BM3", "MATCH (p:Person) WHERE [p.w] = [] RETURN p.name AS n", BOTH),
    ("BM4", "MATCH (p:Person {name: 'Jules'}) RETURN {w: p.w, age: p.age} AS m", BOTH),
    ("BM5", "MATCH (p:Person {name: 'Vincent'}) OPTIONAL MATCH (p)-[:LIVES_IN]->(c:City) RETURN [c.name, p][1].name AS n", BOTH),
    ("BM6", "MATCH (p:Person {name: 'Jules'}) RETURN 3 IN [p.w, 19] AS i, 19 IN [p.w, 19] AS j", BOTH),
    ("BM7", "MATCH p = (a:Person {name: 'Jules'})-[:KNOWS]->(b)-[:KNOWS]->?(c) RETURN length(p) AS l, b.name AS b, c.name AS c", GQL),
]:
    case(case_id, query, languages)

# BN: edge types. An edge has one type, so a second `:` is a syntax error that names the
#     quoted type and the alternatives (BN1, BN2: the KNOWS or LIVES_IN edges came back),
#     and so is a conjunction in GQL (BN4); openCypher 9's `:A|:B` writes alternatives
#     (BN3; it failed with "Expected identifier"). An EXISTS whose path ends at a node of
#     the row searches from its start (BN5, BN6: the same rows as the semi-join)
for case_id, query, languages in [
    ("BN1", "MATCH (a:Person)-[:KNOWS:LIVES_IN]->(b) RETURN a.name AS a, b.name AS b", BOTH),
    ("BN2", "MATCH (a:Person {name: 'Alix'})-[:KNOWS:LIVES_IN*1..2]->(b) RETURN DISTINCT b.name AS b", BOTH),
    ("BN3", "MATCH (a:Person)-[:KNOWS|:LIVES_IN]->(b) RETURN a.name AS a, b.name AS b", CYPHER),
    ("BN4", "MATCH (a:Person)-[:KNOWS&LIVES_IN]->(b) RETURN b.name AS b", GQL),
    ("BN5", "MATCH (a:Person {name: 'Alix'}), (b:Person) WHERE EXISTS { MATCH (a)-[:KNOWS*1..3]->(b) } RETURN b.name AS b", BOTH),
    ("BN6", "MATCH (a:Person), (b:Person) WHERE EXISTS { MATCH (a)-[:KNOWS*1..4]-(b) } RETURN a.name AS a, count(b) AS reached", GQL),
]:
    case(case_id, query, languages)

# BO: GQL has `=~`, Cypher's whole-string regular expression match (BO1 to BO3; GQL
#     refused it with "Expected expression"). A pattern that is not a regular
#     expression is an error that names it (BO4; Cypher matched nothing). A call to an
#     unknown function is an error before any row is read (BO5, BO6, BO8; it was null
#     for every row, so the WHERE of BO6 kept none). BO7 is the control: function names
#     in any case still run
for case_id, query, languages in [
    ("BO1", "MATCH (p:Person) WHERE p.name =~ '(A|G).*' RETURN p.name AS n", BOTH),
    ("BO2", "MATCH (p:Person) WHERE NOT p.name =~ '.*(lix|nce).*' RETURN p.name AS n", BOTH),
    ("BO3", "MATCH (p:Person) RETURN p.name AS n, p.name =~ '[A-J].*' AS early", BOTH),
    ("BO4", "MATCH (p:Person) WHERE p.name =~ '(Alix' RETURN p.name AS n", BOTH),
    ("BO5", "MATCH (p:Person) RETURN no_such_function(p.name) AS x", BOTH),
    ("BO6", "MATCH (p:Person) WHERE upperr(p.name) = 'ALIX' RETURN p.name AS n", BOTH),
    ("BO7", "MATCH (p:Person) RETURN toUpper(p.name) AS u, UPPER(p.name) AS v, Size(p.name) AS s", BOTH),
    ("BO8", "MATCH (p:Person) RETURN [x IN [p.name] | lowerr(x)] AS l", BOTH),
]:
    case(case_id, query, languages)

# AS: zoned datetimes compare by their instant with every ordering operator, also against
#     a timestamp (they gave null, and `=` against a timestamp was false); min and max of
#     zoned values are the earliest and latest instant (they returned the first value)
for case_id, query in [
    ("AS1", "RETURN zoned_datetime('2022-01-01T00:00:00+01:00') < zoned_datetime('2021-12-31T23:30:00Z') AS lt, zoned_datetime('2022-01-01T00:00:00+01:00') >= zoned_datetime('2021-12-31T23:00:00Z') AS ge"),
    ("AS2", "RETURN datetime('2024-03-15T10:30:00Z') = zoned_datetime('2024-03-15T11:30:00+01:00') AS eq, datetime('2024-03-15T10:30:00Z') < zoned_datetime('2024-03-15T11:31:00+01:00') AS lt"),
    ("AS3", "UNWIND [zoned_datetime('2024-03-15T10:30:00+01:00'), zoned_datetime('2019-03-15T10:30:00Z')] AS x RETURN min(x) = zoned_datetime('2019-03-15T10:30:00Z') AS lo, max(x) = zoned_datetime('2024-03-15T09:30:00Z') AS hi"),
]:
    case(case_id, query, BOTH, "empty")

# AT: GQL puts nulls last by default in both directions; Cypher keeps them first when
#     descending (openCypher); a top-K keeps the largest values in GQL
for case_id, query in [
    ("AT1", "MATCH (n:Person) RETURN n.name AS name, n.w AS w ORDER BY w DESC, name"),
    ("AT2", "MATCH (n:Person) RETURN n.name AS name ORDER BY n.w DESC LIMIT 3"),
]:
    ordered(case_id, query, BOTH)

# AU: a call of an unknown function, or with a number of arguments the function does not
#     take, fails with its name (it gave null); path_length(p) is the number of edges
for case_id, query in [
    ("AU1", "RETURN nope(1) AS v"),
    ("AU2", "MATCH (n:Person) WHERE toStringOrNull(n.age) = '30' RETURN n.name AS n"),
    ("AU3", "RETURN toUpper('a', 'b') AS v"),
    ("AU4", "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN b.name AS b, path_length(p) AS hops"),
]:
    case(case_id, query, BOTH)

# BP: the variables a CALL subquery imports stay in scope after a WITH in its body that
#     leaves them out (#545; it failed with "Undefined variable"): a scope clause, Cypher's
#     importing WITH (BP3), GQL's import of the whole row (BP4), an aggregating WITH whose
#     count over no match is one row (BP2), a WHERE that reads an import (BP5, BP6), an
#     EXISTS that matches from it (BP7: it matched from any node), an imported edge (BP8)
#     and value (BP13), nested and OPTIONAL calls (BP9, BP10), ORDER BY and LIMIT (BP11), a
#     UNION in the body (BP12). BP14: a WITH that binds the import's name to a value of its
#     own; BP15: a variable that is not imported stays out of scope
for case_id, query, languages in [
    ("BP1", "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 1 AS x RETURN a.name AS n, a.w AS w, x } RETURN n, w, x", BOTH),
    ("BP2", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k RETURN a.name AS n, a.w AS w, k } RETURN n, w, k", BOTH),
    ("BP3", "MATCH (a:Person) CALL { WITH a MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k RETURN a.name AS n, k } RETURN n, k", CYPHER),
    ("BP4", "MATCH (a:Person) CALL { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k RETURN a.name AS n, k } RETURN n, k", GQL),
    ("BP5", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH b WHERE b.age > a.age RETURN b.name AS older } RETURN a.name AS a, older", BOTH),
    ("BP6", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k WHERE k = 0 OR a.age > 35 RETURN a.name AS n, k } RETURN n, k", BOTH),
    ("BP7", "MATCH (a:Person) CALL (a) { WITH 1 AS x RETURN EXISTS { MATCH (a)-[:LIVES_IN]->(:City) } AS housed } RETURN a.name AS n, housed", BOTH),
    ("BP8", "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) CALL (r) { MATCH (c:City) WITH count(c) AS k RETURN type(r) AS t, r.w AS w, k } RETURN t, w, k", BOTH),
    ("BP9", "MATCH (a:Person) CALL (a) { CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH count(b) AS k RETURN k } WITH k + 1 AS s RETURN a.name AS n, s } RETURN n, s", BOTH),
    ("BP10", "MATCH (a:Person) OPTIONAL CALL (a) { MATCH (a)-[:LIVES_IN]->(c) WITH c RETURN a.name || ' in ' || c.name AS s } RETURN a.name AS a, s", GQL),
    ("BP11", "MATCH (a:Person) CALL (a) { MATCH (a)-[:KNOWS]->(b) WITH b ORDER BY b.age DESC LIMIT 1 RETURN a.name AS n, b.name AS f } RETURN n, f", CYPHER),
    ("BP12", "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 1 AS x RETURN a.name AS n, x UNION ALL WITH 2 AS x RETURN a.name AS n, x } RETURN n, x", BOTH),
    ("BP13", "UNWIND [3, 19] AS v CALL (v) { MATCH (p:Person) WITH count(p) AS k RETURN v + k AS s } RETURN s", BOTH),
    ("BP14", "MATCH (a:Person {name: 'Alix'}) CALL (a) { WITH 3 AS a RETURN a AS v } RETURN v", BOTH),
    ("BP15", "MATCH (a:Person {name: 'Alix'}), (c:City {name: 'Paris'}) CALL (a) { WITH 1 AS x RETURN c.name AS m } RETURN m", BOTH),
]:
    case(case_id, query, languages)

# BQ: grouping keys beside an aggregate over a property, an expression or a CASE come
#     back as their values, in the groups count(*) alone gives (#589; they came back as 0
#     and every group merged into one): GQL LET keys with GROUP BY (BQ1 to BQ4; BQ3 has
#     the shape of Microsoft Fabric's multi-column grouping example) and a WITH that
#     names the keys (BQ5, BQ6)
for case_id, query, languages, is_ordered in [
    ("BQ1", "MATCH (p:Person) LET older = p.age > 28 LET weighted = p.w IS NOT NULL RETURN older, weighted, count(*) AS n, avg(p.age) AS a GROUP BY older, weighted", GQL, False),
    ("BQ2", "MATCH (a:Person)-[k:KNOWS]-(b:Person) LET who = a.name RETURN who, count(*) AS n, avg(b.age) AS age, min(k.since) AS first, max(b.w) AS w GROUP BY who", GQL, False),
    ("BQ3", "MATCH (p:Person) LET older = p.age > 28 LET weighted = p.w IS NOT NULL RETURN older, weighted, count(*) AS n, avg(p.age) AS a, min(p.name) AS first, max(p.w) AS w GROUP BY older, weighted ORDER BY a DESC, first LIMIT 10", GQL, True),
    ("BQ4", "MATCH (a:Person)-[k:KNOWS]-(b:Person) LET who = a.name RETURN who, sum(a.age + b.age) AS ages, sum(CASE WHEN b.age > 28 THEN 1 ELSE 0 END) AS older, count(DISTINCT CASE WHEN k.since > 2011 THEN b.name END) AS late GROUP BY who", GQL, False),
    ("BQ5", "MATCH (p:Person) WITH p, p.age > 28 AS older, p.w IS NOT NULL AS weighted RETURN older, weighted, count(*) AS n, avg(p.age) AS a, sum(p.age * 2) AS twice, sum(CASE WHEN p.name < 'K' THEN 1 ELSE 0 END) AS early", BOTH, False),
    ("BQ6", "MATCH (a:Person)-[k:KNOWS]->(b:Person) WITH a.name AS who, b, k RETURN who, count(*) AS n, avg(b.age) AS age, max(k.since) AS last, count(DISTINCT CASE WHEN b.w IS NULL THEN b.name END) AS unweighted", BOTH, False),
]:
    case(case_id, query, languages, "social", is_ordered)

# BR: a statement that ends with a write and no RETURN has no result, no columns and no rows
#     (#580; ISO GQL's omitted result). A GQL statement of one INSERT or CREATE returned the
#     last node it created (BR1 to BR3), also after NEXT (BR4). A lone INSERT after NEXT
#     reads the rows the statement before returns (BR5: it created a new node `w`), and
#     one empty row after a statement without a result (BR7: it wrote nothing). A DELETE
#     of a variable nothing binds fails (BR9: it deleted every node). These cases write,
#     in order, on a database of their own (fixture `writes`); BR6, BR8 and BR10 read what
#     they wrote.
for case_id, query, languages in [
    ("BR1", "CREATE (:W {k: 3})", BOTH),
    ("BR2", "INSERT (:W {k: 19})-[:R]->(:W {k: 88})", GQL),
    ("BR3", "INSERT (:W {k: 3}), (:W {k: 19})", GQL),
    ("BR4", "INSERT (:W {k: 88}) NEXT INSERT (:W {k: 3})", GQL),
    ("BR5", "MATCH (w:W {k: 88}) RETURN w NEXT INSERT (w)-[:R]->(:V {k: 19})", GQL),
    ("BR6", "MATCH (n) OPTIONAL MATCH (n)-[:R]->(m) RETURN labels(n) AS l, n.k AS k, m.k AS to", BOTH),
    ("BR7", "MATCH (w:W {k: 3}) SET w.k = 3 NEXT INSERT (:U {k: 88})", GQL),
    ("BR8", "MATCH (u:U) RETURN count(u) AS n", BOTH),
    ("BR9", "DETACH DELETE w", GQL),
    ("BR10", "MATCH (n) RETURN count(n) AS n", BOTH),
]:
    case(case_id, query, languages, "writes")

# BS: a mistake in the query text is a semantic error that names it (#588), never an
#     internal error: an unknown procedure (BS1), an unknown YIELD column (BS2), a
#     function called with no argument (BS3), a Cypher DELETE of a variable nothing
#     binds (BS4, which said only "DELETE requires input"; a write, so on the `writes`
#     fixture after the BR cases).
for case_id, query, languages in [
    ("BS1", "CALL grafeo.vincent()", BOTH),
    ("BS2", "CALL grafeo.procedures() YIELD mia", GQL),
    ("BS3", "MATCH (p:Person)-[r]->(q) RETURN type()", BOTH),
]:
    case(case_id, query, languages)
case("BS4", "DELETE w", CYPHER, "writes")

# BT: a GQL GROUP BY names an alias of the RETURN list, as Microsoft Fabric documents it
#     (it was an undefined variable): a computed key (BT1), an alias and an expression at
#     once in the shape of Fabric's example (BT2), an alias key after an aggregate item,
#     with ORDER BY (BT3), the alias of a node (BT4). A RETURN item that is neither a
#     grouping key nor an aggregate says so (BT5; it was "Undefined variable 'p.name'").
#     A RETURN item over a grouped node or key is computed per group (BT6, BT7; they were
#     undefined variables), and one that reads another variable of the grouped rows
#     beside an aggregate says so (BT8; it was null in every group)
for case_id, query, is_ordered in [
    ("BT1", "MATCH (p:Person) RETURN p.age % 2 AS odd, count(*) AS n GROUP BY odd", False),
    ("BT2", "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN c.w AS cityW, c.name, count(*) AS population, avg(p.age) AS average_age GROUP BY cityW, c.name", False),
    ("BT3", "MATCH (a:Person)-[k:KNOWS]-(b:Person) RETURN count(*) AS n, a.name AS who, max(k.since) AS last GROUP BY who ORDER BY who", True),
    ("BT4", "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b AS friend, count(*) AS n GROUP BY friend", False),
    ("BT5", "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name AS name, count(*) AS n GROUP BY c", False),
    ("BT6", "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN c.name AS city, count(*) AS n GROUP BY c", False),
    ("BT7", "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN upper(a.name) AS who, count(*) * 100 + a.age AS code GROUP BY a", False),
    ("BT8", "MATCH (p:Person) RETURN p.name AS name, count(*) + p.age AS n GROUP BY p.name", False),
]:
    case(case_id, query, GQL, "social", is_ordered)

# fmt: on
