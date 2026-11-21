//! Selective path patterns over several edge patterns (ISO/IEC 39075:2024 16.6): a
//! search prefix (`ANY`, `ANY k`, `ANY SHORTEST`, `SHORTEST k`, `ALL SHORTEST`,
//! `SHORTEST k GROUPS`) applies to the whole path pattern, a linear chain of fixed and
//! quantified edge patterns. The matches are partitioned by their first and last node,
//! each partition is selected by the prefix, and the length of a path counts every edge,
//! fixed ones included. The node patterns between the edge patterns keep their labels,
//! properties and `WHERE` at their place, and every variable of the pattern binds.
//!
//! The oracle of each case is the same pattern without the prefix, every quantifier
//! bounded (at the number of nodes times the number of quantified edge patterns, so that
//! every shortest path is among its matches), grouped by endpoint pair and selected
//! here: `ANY SHORTEST` gives one of the shortest paths, `ALL SHORTEST` all of them,
//! `SHORTEST k` k paths of the k smallest lengths, `SHORTEST k GROUPS` every path of the
//! k smallest lengths, `ANY k` k matches. A match is a path with its variables, so two
//! matches of one path that split it differently between two quantified edge patterns
//! are two matches, in the oracle as in the search.
//!
//! graph-bench's SQL/PGQ IS6 runs verbatim, and SQL/PGQ answers as GQL does for the same
//! `MATCH`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main selective_path_patterns::
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use grafeo_common::types::Value;
use grafeo_common::utils::error::{ErrorCode, LimitBound, LimitRefusal};
use grafeo_engine::{Config, GrafeoDB};

// --- Graphs -------------------------------------------------------------------------

/// A node of a test graph: its name and its labels (`":Person"`, `":Post:Stop"`).
type Node<'a> = (&'a str, &'a str);

/// An edge of a test graph: from, type, to, its key `k` (which names it in the
/// results), and more properties (`""` or `", w: 2"`).
type Edge<'a> = (&'a str, &'a str, &'a str, &'a str, &'a str);

/// Inserts `nodes` and then `edges`, in order (the order of the store's adjacency).
fn fill(db: &GrafeoDB, nodes: &[Node], edges: &[Edge]) {
    for (name, labels) in nodes {
        db.execute(&format!("INSERT ({labels} {{name: '{name}'}})"))
            .unwrap();
    }
    for (from, kind, to, key, more) in edges {
        db.execute(&format!(
            "MATCH (x {{name: '{from}'}}), (y {{name: '{to}'}}) \
             INSERT (x)-[:{kind} {{k: '{key}'{more}}}]->(y)"
        ))
        .unwrap();
    }
}

fn graph(nodes: &[Node], edges: &[Edge]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    fill(&db, nodes, edges);
    db
}

// --- Results ------------------------------------------------------------------------

/// A value as text: strings bare, lists in brackets.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Float64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(text).collect::<Vec<_>>().join(", ")
        ),
        other => panic!("unexpected value {other:?}"),
    }
}

/// The rows of the GQL `query` as text, sorted.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    rows.sort();
    rows
}

/// `want` as sorted rows of text.
fn expected(want: &[&[&str]]) -> Vec<Vec<String>> {
    let mut want: Vec<Vec<String>> = want
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    want.sort();
    want
}

fn assert_rows(db: &GrafeoDB, query: &str, want: &[&[&str]]) {
    assert_eq!(rows(db, query), expected(want), "GQL: {query}");
}

/// A path: the names of its nodes and the keys of its edges, in path order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Path {
    nodes: Vec<String>,
    edges: Vec<String>,
}

impl Path {
    fn pair(&self) -> (String, String) {
        (
            self.nodes.first().cloned().unwrap_or_default(),
            self.nodes.last().cloned().unwrap_or_default(),
        )
    }
}

/// The paths of each pair of endpoints, each list sorted by length, then by path.
type Partitions = BTreeMap<(String, String), Vec<Path>>;

/// The paths `MATCH p = <prefix> <pattern>` binds, in the order of the rows.
fn paths(db: &GrafeoDB, prefix: &str, pattern: &str) -> Vec<Path> {
    let query = format!(
        "MATCH p = {prefix} {pattern} \
         RETURN [n IN nodes(p) | n.name] AS nodes, [e IN edges(p) | e.k] AS edges"
    );
    let result = db
        .execute(&query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    result
        .rows()
        .iter()
        .map(|row| {
            let names = |value: &Value| match value {
                Value::List(items) => items.iter().map(text).collect(),
                other => panic!("{query}: a list, got {other:?}"),
            };
            Path {
                nodes: names(&row[0]),
                edges: names(&row[1]),
            }
        })
        .collect()
}

fn partitions(paths: Vec<Path>) -> Partitions {
    let mut partitions = Partitions::new();
    for path in paths {
        partitions.entry(path.pair()).or_default().push(path);
    }
    for paths in partitions.values_mut() {
        paths.sort_by(|a, b| a.edges.len().cmp(&b.edges.len()).then_with(|| a.cmp(b)));
    }
    partitions
}

/// A selection of a search prefix.
#[derive(Debug, Clone, Copy)]
enum Selection {
    /// `ANY` (1) and `ANY k`.
    Any(usize),
    /// `ANY SHORTEST` (1) and `SHORTEST k`.
    Shortest(usize),
    /// `ALL SHORTEST` (1) and `SHORTEST k GROUPS`.
    Groups(usize),
}

use Selection::{Any, Groups, Shortest};

/// Every selection, with one and with two paths or groups.
const SELECTIONS: [Selection; 6] = [
    Any(1),
    Any(2),
    Shortest(1),
    Shortest(2),
    Groups(1),
    Groups(2),
];

impl Selection {
    fn prefix(self) -> String {
        self.with_mode("")
    }

    /// The prefix with the path mode `mode` (`""` for none): before `GROUPS`.
    fn with_mode(self, mode: &str) -> String {
        match self {
            Any(1) => format!("ANY {mode}"),
            Any(k) => format!("ANY {k} {mode}"),
            Shortest(1) => format!("ANY SHORTEST {mode}"),
            Shortest(k) => format!("SHORTEST {k} {mode}"),
            Groups(1) => format!("ALL SHORTEST {mode}"),
            Groups(k) => format!("SHORTEST {k} {mode} GROUPS"),
        }
    }
}

/// Removes one `path` from `from`, which must hold it.
fn take(from: &mut Vec<Path>, path: &Path, what: &str) {
    let index = from
        .iter()
        .position(|candidate| candidate == path)
        .unwrap_or_else(|| panic!("{what}: {path:?} is no match of its pair (or one too many)"));
    from.remove(index);
}

/// The lengths of `paths`, in order.
fn lengths(paths: &[Path]) -> Vec<usize> {
    paths.iter().map(|path| path.edges.len()).collect()
}

/// Checks `MATCH p = <selection with mode> <pattern>` against the matches of `bounded`
/// (the same pattern with every quantifier bounded) under `mode`, the path mode (`""`
/// for WALK).
fn check(db: &GrafeoDB, selection: Selection, mode: &str, pattern: &str, bounded: &str) {
    let prefix = selection.with_mode(mode);
    let what = format!("{prefix} {pattern}");
    let oracle = partitions(paths(db, mode, bounded));
    let got = partitions(paths(db, &prefix, pattern));
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        oracle.keys().collect::<Vec<_>>(),
        "{what}: the pairs of endpoints"
    );
    for (pair, selected) in &got {
        let all = &oracle[pair];
        let what = format!("{what}, pair {pair:?}");
        // Every path selected is a match of its pair, once per match
        let mut matches = all.clone();
        for path in selected {
            take(&mut matches, path, &what);
        }
        match selection {
            Any(k) => assert_eq!(selected.len(), k.min(all.len()), "{what}: {selected:?}"),
            Shortest(k) => {
                assert_eq!(
                    lengths(selected),
                    lengths(&all[..k.min(all.len())]),
                    "{what}: the k smallest lengths, {selected:?} of {all:?}"
                );
            }
            Groups(k) => {
                let mut kept: Vec<usize> = lengths(all);
                kept.dedup();
                kept.truncate(k);
                let want: Vec<Path> = all
                    .iter()
                    .filter(|path| kept.contains(&path.edges.len()))
                    .cloned()
                    .collect();
                assert_eq!(
                    selected, &want,
                    "{what}: every path of the k smallest lengths"
                );
            }
        }
    }
    // The same text picks the same paths again
    if matches!(selection, Shortest(_) | Any(_)) {
        let again = partitions(paths(db, &prefix, pattern));
        assert_eq!(got, again, "{what}: the same paths on a second run");
    }
}

/// [`check`] under every selection, in WALK mode.
fn check_all(db: &GrafeoDB, pattern: &str, bounded: &str) {
    for selection in SELECTIONS {
        check(db, selection, "", pattern, bounded);
    }
}

// --- 1, 6, 19: graph-bench's IS6 on an SNB-shaped graph -------------------------------

/// graph-bench's SQL/PGQ IS6 (`graph_bench/suites/snb_sql.py`), verbatim.
const IS6_SQL_PGQ: &str = "\
SELECT forumId, forumTitle, moderatorId, moderatorFirstName, moderatorLastName
FROM GRAPH_TABLE (snb
  MATCH ANY SHORTEST (m:Post|Comment)-[r:REPLY_OF]->*(p:Post)
        <-[co:CONTAINER_OF]-(f:Forum)-[hm:HAS_MODERATOR]->(mod:Person)
  WHERE m.id = $messageId
  COLUMNS (f.id AS forumId, f.title AS forumTitle, mod.id AS moderatorId,
           mod.firstName AS moderatorFirstName, mod.lastName AS moderatorLastName)
)";

/// The same `MATCH` in GQL.
const IS6_GQL: &str = "\
MATCH ANY SHORTEST (m:Post|Comment)-[r:REPLY_OF]->*(p:Post)
      <-[co:CONTAINER_OF]-(f:Forum)-[hm:HAS_MODERATOR]->(mod:Person)
WHERE m.id = $messageId
RETURN f.id AS forumId, f.title AS forumTitle, mod.id AS moderatorId,
       mod.firstName AS moderatorFirstName, mod.lastName AS moderatorLastName";

/// The moderators (id, first name, last name), the forums (id, title, moderator), the
/// posts (id, forum) and the comments (id, the message it replies to) of the SNB-shaped
/// graph: two forums, a post in each, and threads of replies, one of them three deep.
const PERSONS: [(i64, &str, &str); 2] = [(1, "Mia", "Wallace"), (2, "Hans", "Landa")];
const FORUMS: [(i64, &str, i64); 2] = [(10, "Wall of Mia", 1), (20, "Wall of Hans", 2)];
const POSTS: [(i64, i64); 2] = [(100, 10), (200, 20)];
const COMMENTS: [(i64, i64); 5] = [(101, 100), (102, 101), (103, 102), (104, 100), (201, 200)];

/// The SNB-shaped graph in the graph `snb` of a new database. Every edge has a key `k`.
fn snb() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH snb").unwrap();
    let snb = db.graph("snb").unwrap();
    let run = |statement: String| {
        snb.execute(&statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    };
    for (id, first, last) in PERSONS {
        run(format!(
            "INSERT (:Person {{id: {id}, firstName: '{first}', lastName: '{last}'}})"
        ));
    }
    for (id, title, moderator) in FORUMS {
        run(format!("INSERT (:Forum {{id: {id}, title: '{title}'}})"));
        run(format!(
            "MATCH (f:Forum {{id: {id}}}), (p:Person {{id: {moderator}}}) \
             INSERT (f)-[:HAS_MODERATOR {{k: 'hm{id}'}}]->(p)"
        ));
    }
    for (id, forum) in POSTS {
        run(format!("INSERT (:Post {{id: {id}}})"));
        run(format!(
            "MATCH (f:Forum {{id: {forum}}}), (p:Post {{id: {id}}}) \
             INSERT (f)-[:CONTAINER_OF {{k: 'co{id}'}}]->(p)"
        ));
    }
    for (id, parent) in COMMENTS {
        run(format!("INSERT (:Comment {{id: {id}}})"));
        run(format!(
            "MATCH (c:Comment {{id: {id}}}), (m {{id: {parent}}}) \
             INSERT (c)-[:REPLY_OF {{k: 'r{id}'}}]->(m)"
        ));
    }
    db
}

/// graph-bench's reference answer of IS6 for `message`: the forum of the post at the
/// root of its thread, and the forum's moderator.
fn is6_reference(message: i64) -> Vec<Value> {
    let mut root = message;
    while let Some(&(_, parent)) = COMMENTS.iter().find(|(id, _)| *id == root) {
        root = parent;
    }
    let (_, forum) = POSTS.iter().find(|(id, _)| *id == root).copied().unwrap();
    let (_, title, moderator) = FORUMS
        .iter()
        .find(|(id, ..)| *id == forum)
        .copied()
        .unwrap();
    let (_, first, last) = PERSONS
        .iter()
        .find(|(id, ..)| *id == moderator)
        .copied()
        .unwrap();
    vec![
        Value::Int64(forum),
        Value::String(title.into()),
        Value::Int64(moderator),
        Value::String(first.into()),
        Value::String(last.into()),
    ]
}

/// The messages of the SNB-shaped graph.
fn messages() -> impl Iterator<Item = i64> {
    POSTS
        .iter()
        .map(|(id, _)| *id)
        .chain(COMMENTS.iter().map(|(id, _)| *id))
}

/// Row 1: graph-bench's SQL/PGQ IS6 runs verbatim and gives graph-bench's reference
/// answer for every message; the same MATCH in GQL gives the same rows.
#[cfg(feature = "sql-pgq")]
#[test]
fn graph_bench_is6_verbatim_in_sql_pgq_and_gql() {
    let db = snb();
    let snb = db.graph("snb").unwrap();
    for message in messages() {
        let params =
            std::collections::HashMap::from([("messageId".to_string(), Value::Int64(message))]);
        let sql = db
            .execute_language(IS6_SQL_PGQ, "sql-pgq", Some(params.clone()))
            .unwrap_or_else(|error| panic!("SQL/PGQ IS6 {message}: {error}"))
            .rows()
            .to_vec();
        assert_eq!(sql, [is6_reference(message)], "SQL/PGQ IS6 {message}");
        let gql = snb
            .execute_with_params(IS6_GQL, params)
            .unwrap_or_else(|error| panic!("GQL IS6 {message}: {error}"))
            .rows()
            .to_vec();
        assert_eq!(gql, sql, "GQL IS6 {message}");
    }
}

/// Rows 1, 6 and 19: per pair of a message and a moderator, IS6's search binds every
/// variable at its place (`r` the list of the reply edges, empty for a post, which is
/// its own root), the same as the oracle: the pattern with `r` bounded and no prefix.
#[test]
fn is6_binds_every_variable_and_agrees_with_the_oracle() {
    let db = snb();
    let snb = db.graph("snb").unwrap();
    let returned = "RETURN m.id AS m, [e IN r | e.k] AS r, p.id AS p, co.k AS co, f.id AS f, \
                    hm.k AS hm, mod.id AS mod, length(path) AS len";
    let selected = |prefix: &str, quantifier: &str| {
        let query = format!(
            "MATCH path = {prefix} (m:Post|Comment)-[r:REPLY_OF]->{quantifier}(p:Post)\
             <-[co:CONTAINER_OF]-(f:Forum)-[hm:HAS_MODERATOR]->(mod:Person) {returned}"
        );
        let mut rows: Vec<Vec<String>> = snb
            .execute(&query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .iter()
            .map(|row| row.iter().map(text).collect())
            .collect();
        rows.sort();
        rows
    };
    // Every message has one path to its moderator; 7 messages, so at most 7 replies
    let search = selected("ANY SHORTEST", "*");
    assert_eq!(search, selected("", "{0,7}"), "the oracle");
    assert_eq!(search, selected("ALL SHORTEST", "*"), "ALL SHORTEST");
    let want: Vec<Vec<String>> = expected(&[
        &["100", "[]", "100", "co100", "10", "hm10", "1", "2"],
        &["101", "[r101]", "100", "co100", "10", "hm10", "1", "3"],
        &[
            "102",
            "[r102, r101]",
            "100",
            "co100",
            "10",
            "hm10",
            "1",
            "4",
        ],
        &[
            "103",
            "[r103, r102, r101]",
            "100",
            "co100",
            "10",
            "hm10",
            "1",
            "5",
        ],
        &["104", "[r104]", "100", "co100", "10", "hm10", "1", "3"],
        &["200", "[]", "200", "co200", "20", "hm20", "2", "2"],
        &["201", "[r201]", "200", "co200", "20", "hm20", "2", "3"],
    ]);
    assert_eq!(search, want);
}

// --- 2 to 5, 7: where the quantified edge patterns are --------------------------------

/// Row 2: a quantified edge pattern between two fixed ones, over a cycle of `S` edges
/// (Gus, Jules, Mia), with ties.
#[test]
fn a_quantified_edge_between_fixed_edges() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
            ("Django", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Alix", "R", "Vincent", "r2", ""),
            ("Gus", "S", "Jules", "s1", ""),
            ("Jules", "S", "Mia", "s2", ""),
            ("Vincent", "S", "Mia", "s3", ""),
            ("Mia", "S", "Gus", "s4", ""),
            ("Mia", "T", "Butch", "t1", ""),
            ("Jules", "T", "Butch", "t2", ""),
            ("Gus", "T", "Django", "t3", ""),
            ("Vincent", "T", "Django", "t4", ""),
        ],
    );
    check_all(
        &db,
        "(a)-[:R]->(b)-[:S]->*(c)-[:T]->(d)",
        "(a)-[:R]->(b)-[:S]->{0,7}(c)-[:T]->(d)",
    );
    check_all(
        &db,
        "(a)-[:R]->(b)-[:S]->+(c)-[:T]->(d)",
        "(a)-[:R]->(b)-[:S]->{1,7}(c)-[:T]->(d)",
    );
}

/// Row 3: a quantified edge pattern after two fixed ones, ending in a cycle of `T`
/// edges (Mia, Butch).
#[test]
fn a_quantified_edge_after_fixed_edges() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "S", "Vincent", "s1", ""),
            ("Gus", "S", "Jules", "s2", ""),
            ("Vincent", "T", "Mia", "t1", ""),
            ("Jules", "T", "Mia", "t2", ""),
            ("Mia", "T", "Butch", "t3", ""),
            ("Butch", "T", "Mia", "t4", ""),
            ("Vincent", "T", "Butch", "t5", ""),
        ],
    );
    check_all(
        &db,
        "(a)-[:R]->(b)-[:S]->(c)-[:T]->+(d)",
        "(a)-[:R]->(b)-[:S]->(c)-[:T]->{1,6}(d)",
    );
}

/// Row 4: fixed edge patterns only, two and three of them: every match is a candidate,
/// so `ALL SHORTEST` returns every match of each pair and `ANY SHORTEST` one.
#[test]
fn fixed_edges_only() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Alix", "R", "Vincent", "r2", ""),
            ("Gus", "S", "Jules", "s1", ""),
            ("Vincent", "S", "Jules", "s2", ""),
            ("Gus", "S", "Mia", "s3", ""),
            ("Jules", "T", "Mia", "t1", ""),
            ("Mia", "T", "Jules", "t2", ""),
        ],
    );
    for pattern in [
        "(a)-[:R]->(b)-[:S]->(c)",
        "(a)-[:R]->(b)-[:S]->(c)-[:T]->(d)",
    ] {
        check_all(&db, pattern, pattern);
        let every = partitions(paths(&db, "", pattern));
        assert_eq!(
            partitions(paths(&db, "ALL SHORTEST", pattern)),
            every,
            "{pattern}"
        );
    }
    // Alix to Jules: through Gus and through Vincent
    assert_eq!(
        paths(
            &db,
            "ALL SHORTEST",
            "(a {name: 'Alix'})-[:R]->(b)-[:S]->(c {name: 'Jules'})"
        )
        .len(),
        2
    );
    assert_eq!(
        paths(
            &db,
            "ANY SHORTEST",
            "(a {name: 'Alix'})-[:R]->(b)-[:S]->(c {name: 'Jules'})"
        )
        .len(),
        1
    );
}

/// Row 5: two quantified edge patterns in one path pattern, of two types and of one
/// type. Of one type, a path splits between them in several ways, each a match.
#[test]
fn two_quantified_edges() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Vincent", "R", "Gus", "r3", ""),
            ("Alix", "S", "Jules", "s1", ""),
            ("Gus", "S", "Jules", "s2", ""),
            ("Vincent", "S", "Mia", "s3", ""),
            ("Jules", "S", "Mia", "s4", ""),
            ("Mia", "S", "Jules", "s5", ""),
        ],
    );
    check_all(
        &db,
        "(a)-[:R]->*(b)-[:S]->+(c)",
        "(a)-[:R]->{0,10}(b)-[:S]->{1,10}(c)",
    );
    check_all(
        &db,
        "(a {name: 'Alix'})-[:R]->*(b)-[:R]->+(c)",
        "(a {name: 'Alix'})-[:R]->{0,10}(b)-[:R]->{1,10}(c)",
    );
    // Alix to Vincent over two R edges: the first pattern takes none or one of them
    assert_rows(
        &db,
        "MATCH ALL SHORTEST (a {name: 'Alix'})-[x:R]->*(b)-[y:R]->+(c {name: 'Vincent'}) \
         RETURN [e IN x | e.k] AS x, [e IN y | e.k] AS y",
        &[&["[]", "[r1, r2]"], &["[r1]", "[r2]"]],
    );
    // The node between the two is a variable of each binding: named and returned, the
    // two splits are two rows
    assert_rows(
        &db,
        "MATCH ALL SHORTEST (a {name: 'Alix'})-[x:R]->*(m)-[y:R]->+(c {name: 'Vincent'}) \
         RETURN m.name, [e IN x | e.k] AS x, [e IN y | e.k] AS y",
        &[&["Alix", "[]", "[r1, r2]"], &["Gus", "[r1]", "[r2]"]],
    );
    // The bindings by the split node and the size of each run (the unselected match
    // reads a property of a run's one edge as null, a fault of its own)
    let returned = "m.name, size(x), size(y)";
    check_bindings(
        &db,
        "(a)-[x:R]->*(m)-[y:S]->+(c)",
        "(a)-[x:R]->{0,10}(m)-[y:S]->{1,10}(c)",
        returned,
    );
    check_bindings(
        &db,
        "(a {name: 'Alix'})-[x:R]->*(m)-[y:R]->+(c)",
        "(a {name: 'Alix'})-[x:R]->{0,10}(m)-[y:R]->{1,10}(c)",
        returned,
    );
}

/// Checks `ALL SHORTEST <pattern>` against the matches of `bounded` as bindings: per
/// pair of endpoints, the multiset of the shortest matches with their nodes, edges and
/// the `returned` expressions (the variables inside the path).
fn check_bindings(db: &GrafeoDB, pattern: &str, bounded: &str, returned: &str) {
    let bindings = |prefix: &str, pattern: &str| -> BTreeMap<(String, String), Vec<Vec<String>>> {
        let query = format!(
            "MATCH p = {prefix} {pattern} RETURN [n IN nodes(p) | n.name][0] AS first, \
             [n IN nodes(p) | n.name][length(p)] AS last, length(p) AS len, \
             [n IN nodes(p) | n.name] AS nodes, [e IN edges(p) | e.k] AS edges, {returned}"
        );
        let mut partitions: BTreeMap<(String, String), Vec<Vec<String>>> = BTreeMap::new();
        for row in rows(db, &query) {
            partitions
                .entry((row[0].clone(), row[1].clone()))
                .or_default()
                .push(row[2..].to_vec());
        }
        partitions
    };
    let mut oracle = bindings("", bounded);
    for matches in oracle.values_mut() {
        let shortest = matches
            .iter()
            .map(|row| row[0].parse::<usize>().unwrap())
            .min()
            .unwrap();
        matches.retain(|row| row[0] == shortest.to_string());
        matches.sort();
    }
    let mut got = bindings("ALL SHORTEST", pattern);
    for matches in got.values_mut() {
        matches.sort();
    }
    assert_eq!(
        got, oracle,
        "ALL SHORTEST {pattern}: the multiset of bindings"
    );
}

/// Row 7: a bounded quantifier between fixed edge patterns: its run has 2 or 3 edges,
/// over a chain of five `S` edges.
#[test]
fn a_bounded_quantifier_between_fixed_edges() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
            ("Django", ":Person"),
            ("Hans", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "S", "Vincent", "s1", ""),
            ("Vincent", "S", "Jules", "s2", ""),
            ("Jules", "S", "Mia", "s3", ""),
            ("Mia", "S", "Butch", "s4", ""),
            ("Butch", "S", "Django", "s5", ""),
            ("Gus", "T", "Hans", "t0", ""),
            ("Vincent", "T", "Hans", "t1", ""),
            ("Jules", "T", "Hans", "t2", ""),
            ("Mia", "T", "Hans", "t3", ""),
            ("Butch", "T", "Hans", "t4", ""),
            ("Django", "T", "Hans", "t5", ""),
        ],
    );
    let pattern = "(a)-[:R]->(b)-[:S]->{2,3}(c)-[:T]->(d)";
    check_all(&db, pattern, pattern);
    assert_rows(
        &db,
        "MATCH ALL SHORTEST (a {name: 'Alix'})-[:R]->(b)-[s:S]->{2,3}(c)-[t:T]->(d) \
         RETURN size(s), t.k",
        &[&["2", "t2"]],
    );
    assert_rows(
        &db,
        "MATCH SHORTEST 2 GROUPS (a {name: 'Alix'})-[:R]->(b)-[s:S]->{2,3}(c)-[t:T]->(d) \
         RETURN size(s), t.k",
        &[&["2", "t2"], &["3", "t3"]],
    );
}

// --- 8 to 11: ties and selections --------------------------------------------------

/// A post in two forums, both moderated by Mia, and a reply to it; the second forum
/// has two `HAS_MODERATOR` edges to Mia.
fn two_forums() -> GrafeoDB {
    graph(
        &[
            ("Pulp", ":Post"),
            ("Reply", ":Comment"),
            ("Amsterdam", ":Forum"),
            ("Utrecht", ":Forum"),
            ("Mia", ":Person"),
        ],
        &[
            ("Reply", "REPLY_OF", "Pulp", "r1", ""),
            ("Amsterdam", "CONTAINER_OF", "Pulp", "co1", ""),
            ("Utrecht", "CONTAINER_OF", "Pulp", "co2", ""),
            ("Amsterdam", "HAS_MODERATOR", "Mia", "hm1", ""),
            ("Utrecht", "HAS_MODERATOR", "Mia", "hm2", ""),
            ("Utrecht", "HAS_MODERATOR", "Mia", "hm3", ""),
        ],
    )
}

const FORUM_PATTERN: &str = "(m:Comment)-[r:REPLY_OF]->*(p:Post)<-[co:CONTAINER_OF]-(f:Forum)\
                             -[hm:HAS_MODERATOR]->(mod:Person)";

/// Rows 8 and 9: `ALL SHORTEST` keeps the tied paths through two forums, and one path
/// per parallel fixed edge.
#[test]
fn all_shortest_keeps_ties_through_two_nodes_and_parallel_edges() {
    let db = two_forums();
    assert_rows(
        &db,
        &format!("MATCH ALL SHORTEST {FORUM_PATTERN} RETURN f.name, co.k, hm.k"),
        &[
            &["Amsterdam", "co1", "hm1"],
            &["Utrecht", "co2", "hm2"],
            &["Utrecht", "co2", "hm3"],
        ],
    );
    let bounded = FORUM_PATTERN.replace("->*", "->{0,5}");
    check_all(&db, FORUM_PATTERN, &bounded);
    assert_eq!(
        paths(&db, "ANY SHORTEST", FORUM_PATTERN).len(),
        1,
        "one pair, one path"
    );
}

/// Rows 10 and 11: `SHORTEST 3`, `SHORTEST 2 GROUPS`, `ANY` and `ANY 2` where the
/// second-shortest paths differ in their fixed edge: Alix to Mia has two paths of 2
/// edges (r1 t1, r2 t2) and two of 3 (r2 s1 t1, r3 s2 t1).
#[test]
fn k_paths_and_k_groups_differ_in_the_fixed_part() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Alix", "R", "Vincent", "r2", ""),
            ("Alix", "R", "Jules", "r3", ""),
            ("Gus", "T", "Mia", "t1", ""),
            ("Vincent", "T", "Mia", "t2", ""),
            ("Vincent", "S", "Gus", "s1", ""),
            ("Jules", "S", "Gus", "s2", ""),
        ],
    );
    let pattern = "(a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)-[:T]->(d {name: 'Mia'})";
    let bounded = "(a {name: 'Alix'})-[:R]->(b)-[:S]->{0,5}(c)-[:T]->(d {name: 'Mia'})";
    for selection in [Shortest(3), Groups(2), Any(1), Any(2), Any(5), Shortest(5)] {
        check(&db, selection, "", pattern, bounded);
    }
    let edges = |prefix: &str| -> Vec<Vec<String>> {
        let mut edges: Vec<Vec<String>> = paths(&db, prefix, pattern)
            .into_iter()
            .map(|path| path.edges)
            .collect();
        edges.sort();
        edges
    };
    let group = |keys: &[&[&str]]| -> Vec<Vec<String>> {
        let mut edges: Vec<Vec<String>> = keys
            .iter()
            .map(|path| path.iter().map(|key| (*key).to_string()).collect())
            .collect();
        edges.sort();
        edges
    };
    assert_eq!(
        edges("SHORTEST 2 GROUPS"),
        group(&[
            &["r1", "t1"],
            &["r2", "t2"],
            &["r2", "s1", "t1"],
            &["r3", "s2", "t1"]
        ])
    );
    assert_eq!(
        edges("ALL SHORTEST"),
        group(&[&["r1", "t1"], &["r2", "t2"]])
    );
}

// --- 12 to 14: node patterns, directions and edge conditions -----------------------

/// Row 12: a label, a property or a `WHERE` on the node pattern between the edge
/// patterns holds at its place: the shortest run of `R` edges ends at Gus, who is no
/// stop, so the path goes on to Jules (Alix, Gus, Vincent, Jules, Mia).
#[test]
fn a_node_pattern_between_edges_filters_at_its_place() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person:Stop"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Vincent", "R", "Jules", "r3", ""),
            ("Gus", "S", "Mia", "s1", ""),
            ("Jules", "S", "Mia", "s2", ""),
        ],
    );
    db.execute("MATCH (j {name: 'Jules'}) SET j.stop = true")
        .unwrap();
    for stop in ["(b:Stop)", "(b {stop: true})", "(b WHERE b.stop = true)"] {
        let pattern = format!("(a {{name: 'Alix'}})-[:R]->*{stop}-[:S]->(c)");
        let bounded = format!("(a {{name: 'Alix'}})-[:R]->{{0,5}}{stop}-[:S]->(c)");
        check_all(&db, &pattern, &bounded);
        assert_rows(
            &db,
            &format!("MATCH p = ANY SHORTEST {pattern} RETURN b.name, c.name, length(p)"),
            &[&["Jules", "Mia", "4"]],
        );
    }
    // Without the filter, Gus is the end of the shortest run
    assert_rows(
        &db,
        "MATCH p = ANY SHORTEST (a {name: 'Alix'})-[:R]->*(b)-[:S]->(c) \
         RETURN b.name, c.name, length(p)",
        &[&["Gus", "Mia", "2"]],
    );
}

/// Row 13: an edge pattern against the direction (`<-[co]-`) and an undirected one.
#[test]
fn mixed_directions() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Vincent", "C", "Gus", "c1", ""),
            ("Jules", "C", "Gus", "c2", ""),
            ("Vincent", "H", "Mia", "h1", ""),
            ("Butch", "H", "Jules", "h2", ""),
            ("Mia", "R", "Butch", "r2", ""),
            ("Butch", "R", "Alix", "r3", ""),
        ],
    );
    check_all(
        &db,
        "(a)-[:R]->(b)<-[co:C]-(f)-[:H]-(m)",
        "(a)-[:R]->(b)<-[co:C]-(f)-[:H]-(m)",
    );
    check_all(
        &db,
        "(a)-[:R]->+(b)<-[co:C]-(f)-[:H]-(m)",
        "(a)-[:R]->{1,6}(b)<-[co:C]-(f)-[:H]-(m)",
    );
    check_all(
        &db,
        "(a)<-[:R]-*(b)-[:H]-(f)-[:C]->+(m)",
        "(a)<-[:R]-{0,12}(b)-[:H]-(f)-[:C]->{1,12}(m)",
    );
}

/// Row 14: a `WHERE` and a property map on a fixed edge pattern and on a quantified
/// one hold for each edge at their place, before the selection.
#[test]
fn edge_conditions_on_fixed_and_quantified_edges() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ", w: 1"),
            ("Alix", "R", "Vincent", "r2", ", w: 2"),
            ("Gus", "S", "Mia", "s1", ", w: 5"),
            ("Vincent", "S", "Gus", "s2", ", w: 3"),
            ("Vincent", "S", "Jules", "s3", ", w: 1"),
            ("Jules", "S", "Mia", "s4", ", w: 9"),
            ("Mia", "T", "Alix", "t1", ", w: 2"),
            ("Mia", "T", "Jules", "t2", ", w: 1"),
            ("Gus", "T", "Alix", "t3", ", w: 2"),
        ],
    );
    check_all(
        &db,
        "(a)-[x:R WHERE x.w > 1]->(b)-[s:S WHERE s.w > 1]->*(c)-[:T {w: 2}]->(d)",
        "(a)-[x:R WHERE x.w > 1]->(b)-[s:S WHERE s.w > 1]->{0,5}(c)-[:T {w: 2}]->(d)",
    );
    // Alix to Alix: r1 (w 1) is out, so through Vincent and Gus (s2) to Mia (s1)
    assert_rows(
        &db,
        "MATCH p = ANY SHORTEST (a {name: 'Alix'})-[x:R WHERE x.w > 1]->(b)\
         -[s:S WHERE s.w > 1]->*(c)-[:T {w: 2}]->(d {name: 'Alix'}) \
         RETURN [e IN edges(p) | e.k]",
        &[&["[r2, s2, t3]"]],
    );
}

// --- 15 and 16: bound endpoints and no path -----------------------------------------

fn chain() -> GrafeoDB {
    graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Django", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "S", "Vincent", "s1", ""),
            ("Vincent", "S", "Jules", "s2", ""),
            ("Gus", "S", "Jules", "s3", ""),
            ("Jules", "T", "Mia", "t1", ""),
            ("Vincent", "T", "Mia", "t2", ""),
        ],
    )
}

/// Row 15: a source bound by an earlier MATCH, and both endpoints bound: one search
/// per input row, and no row for a pair without a path.
#[test]
fn bound_endpoints_search_per_input_row() {
    let db = chain();
    assert_rows(
        &db,
        "MATCH (a:Person) WHERE a.name IN ['Alix', 'Gus'] \
         MATCH p = ANY SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(d) \
         RETURN a.name, d.name, length(p)",
        &[&["Alix", "Mia", "3"]],
    );
    assert_rows(
        &db,
        "MATCH (a:Person), (d:Person) WHERE a.name IN ['Alix', 'Django'] AND d.name = 'Mia' \
         MATCH p = ALL SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(d) \
         RETURN a.name, d.name, [e IN edges(p) | e.k]",
        &[
            &["Alix", "Mia", "[r1, s1, t2]"],
            &["Alix", "Mia", "[r1, s3, t1]"],
        ],
    );
    // Each input row is a partition of its own: Alix twice, two rows each
    assert_rows(
        &db,
        "UNWIND [1, 2] AS x MATCH (a {name: 'Alix'}), (d {name: 'Mia'}) \
         MATCH ALL SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(d) RETURN x, count(*)",
        &[&["1", "2"], &["2", "2"]],
    );
}

/// Row 16: no path for any pair gives no rows.
#[test]
fn no_path_gives_no_rows() {
    let db = chain();
    for selection in SELECTIONS {
        for pattern in [
            "(a)-[:T]->(b)-[:R]->+(c)",
            "(a {name: 'Django'})-[:R]->(b)-[:S]->*(c)",
            "(a)-[:R]->(b)-[:T]->(c)",
        ] {
            assert_eq!(
                paths(&db, &selection.prefix(), pattern),
                [],
                "{} {pattern}",
                selection.prefix()
            );
        }
    }
}

// --- 17 and 18: cycles, path modes and the path variable ---------------------------

/// Row 17: a self-loop in the fixed part and a cycle back to the start, under each path
/// mode: WALK may repeat, TRAIL repeats no edge, ACYCLIC no node, SIMPLE no node but
/// the first as the last.
#[test]
fn cycles_under_each_path_mode() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "L", "Gus", "l1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Vincent", "R", "Alix", "r3", ""),
        ],
    );
    // A second walk round the cycle of three is three edges longer: the oracle's bound
    // holds the two shortest lengths
    for mode in ["", "WALK", "TRAIL", "ACYCLIC", "SIMPLE"] {
        for selection in SELECTIONS {
            check(
                &db,
                selection,
                mode,
                "(a)-[:R]->(b)-[:L]->(c)-[:R]->+(d)",
                "(a)-[:R]->(b)-[:L]->(c)-[:R]->{1,7}(d)",
            );
            check(
                &db,
                selection,
                mode,
                "(a)-[:R]->(b)-[:R]->+(c)",
                "(a)-[:R]->(b)-[:R]->{1,7}(c)",
            );
        }
    }
    // Alix back to herself: three edges round the cycle, not under ACYCLIC
    let back = "(a {name: 'Alix'})-[:R]->(b)-[:R]->+(c {name: 'Alix'})";
    assert_eq!(paths(&db, "ANY SHORTEST", back).len(), 1);
    assert_eq!(paths(&db, "ANY SHORTEST SIMPLE", back).len(), 1);
    assert_eq!(paths(&db, "ANY SHORTEST TRAIL", back).len(), 1);
    assert_eq!(paths(&db, "ANY SHORTEST ACYCLIC", back), []);
    // Gus to Gus over the self-loop, then round the cycle: a walk, not a trail again
    let round = "(a {name: 'Alix'})-[:R]->(b)-[:L]->(c)-[:R]->+(d {name: 'Gus'})";
    assert_eq!(
        paths(&db, "ANY SHORTEST", round)[0].edges,
        ["r1", "l1", "r2", "r3", "r1"]
    );
    assert_eq!(paths(&db, "ANY SHORTEST TRAIL", round), []);
}

/// Row 18: a path variable holds every node and edge of the path in order, fixed ones
/// included: `length`, `path_length`, `nodes`, `edges` and `relationships`.
#[test]
fn the_path_variable_holds_the_whole_path() {
    let db = chain();
    assert_rows(
        &db,
        "MATCH p = ALL SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)-[:T]->(d) \
         RETURN length(p), path_length(p), [n IN nodes(p) | n.name], \
         [e IN edges(p) | e.k], [e IN relationships(p) | e.k]",
        &[
            &[
                "3",
                "3",
                "[Alix, Gus, Jules, Mia]",
                "[r1, s3, t1]",
                "[r1, s3, t1]",
            ],
            &[
                "3",
                "3",
                "[Alix, Gus, Vincent, Mia]",
                "[r1, s1, t2]",
                "[r1, s1, t2]",
            ],
        ],
    );
    assert_rows(
        &db,
        "MATCH p = ANY SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)-[:T]->(d) \
         RETURN length(p), size(nodes(p)), size(edges(p))",
        &[&["3", "4", "3"]],
    );
    assert_rows(
        &db,
        "MATCH p = ALL SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->(c)-[:T]->(d) \
         RETURN length(p), [n IN nodes(p) | n.name], [e IN edges(p) | e.k]",
        &[
            &["3", "[Alix, Gus, Jules, Mia]", "[r1, s3, t1]"],
            &["3", "[Alix, Gus, Vincent, Mia]", "[r1, s1, t2]"],
        ],
    );
}

// --- 19: every variable binds --------------------------------------------------------

/// Row 19: every variable of the pattern binds at its place: the nodes between the edge
/// patterns, the fixed edges, the quantified edge as the list of its run.
#[test]
fn every_variable_binds_at_its_place() {
    let db = two_forums();
    assert_rows(
        &db,
        &format!(
            "MATCH ALL SHORTEST {FORUM_PATTERN} WHERE f.name = 'Amsterdam' \
             RETURN m.name, [e IN r | e.k], size(r), p.name, co.k, f.name, hm.k, mod.name"
        ),
        &[&[
            "Reply",
            "[r1]",
            "1",
            "Pulp",
            "co1",
            "Amsterdam",
            "hm1",
            "Mia",
        ]],
    );
    // The intermediate variables read as nodes and edges after the search
    assert_rows(
        &db,
        &format!(
            "MATCH ALL SHORTEST {FORUM_PATTERN} \
             RETURN labels(p), labels(f), type(co), type(hm), f.name"
        ),
        &[
            &[
                "[Post]",
                "[Forum]",
                "CONTAINER_OF",
                "HAS_MODERATOR",
                "Amsterdam",
            ],
            &[
                "[Post]",
                "[Forum]",
                "CONTAINER_OF",
                "HAS_MODERATOR",
                "Utrecht",
            ],
            &[
                "[Post]",
                "[Forum]",
                "CONTAINER_OF",
                "HAS_MODERATOR",
                "Utrecht",
            ],
        ],
    );
}

// --- 20: what the transaction sees ---------------------------------------------------

const MIA_FROM_ALIX: &str = "MATCH p = ANY SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)\
                             -[:T]->(d {name: 'Mia'}) RETURN length(p)";

const SHORTCUT: &str = "MATCH (g {name: 'Gus'}), (m {name: 'Mia'}) INSERT (g)-[:T {k: 't9'}]->(m)";

/// Row 20: the search sees its own transaction's uncommitted shortcut, another
/// transaction does not, and a transaction that read before a commit keeps its
/// snapshot.
#[test]
fn the_search_sees_what_its_transaction_sees() {
    let db = chain();
    // Without the shortcut Alix reaches Mia in 3 edges (r1 s1 t2), with it in 2 (r1 t9)
    let without = [vec![Value::Int64(3)]];
    let with = [vec![Value::Int64(2)]];
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute(SHORTCUT).unwrap();
    assert_eq!(writer.execute(MIA_FROM_ALIX).unwrap().rows(), with);
    assert_eq!(db.execute(MIA_FROM_ALIX).unwrap().rows(), without);
    writer.rollback().unwrap();
    assert_eq!(db.execute(MIA_FROM_ALIX).unwrap().rows(), without);

    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(reader.execute(MIA_FROM_ALIX).unwrap().rows(), without);
    db.execute(SHORTCUT).unwrap();
    assert_eq!(reader.execute(MIA_FROM_ALIX).unwrap().rows(), without);
    reader.commit().unwrap();
    assert_eq!(db.execute(MIA_FROM_ALIX).unwrap().rows(), with);
}

// --- 21 and 22: budget and deadline ---------------------------------------------------

/// A ladder from Alix: ten rungs of two parallel `R` edges, then an `S` edge to Mia, so
/// Alix has 2^10 shortest paths to Mia.
fn ladder(db: &GrafeoDB) {
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Mia'})")
        .unwrap();
    db.execute("FOR i IN range(1, 10) INSERT (:Stop {i: i})")
        .unwrap();
    db.execute("MATCH (a {name: 'Alix'}), (s:Stop {i: 1}) INSERT (a)-[:R]->(s), (a)-[:R]->(s)")
        .unwrap();
    db.execute("MATCH (s:Stop), (t:Stop) WHERE t.i = s.i + 1 INSERT (s)-[:R]->(t), (s)-[:R]->(t)")
        .unwrap();
    db.execute("MATCH (s:Stop {i: 10}), (m {name: 'Mia'}) INSERT (s)-[:S]->(m)")
        .unwrap();
}

/// Row 21: past the path search budget the search is refused with GRAFEO-S001 and its
/// statement changes nothing; one path, with the states its search holds, fits.
#[test]
fn past_the_path_search_budget_is_refused() {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(8 * 1024)).unwrap();
    ladder(&db);
    let search = "(a {name: 'Alix'})-[:R]->+(b)-[:S]->(c {name: 'Mia'})";
    let error = db
        .execute(&format!("MATCH ALL SHORTEST {search} INSERT (:Mark)"))
        .expect_err("1,024 paths do not fit in 8 KiB");
    assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
    assert!(
        matches!(
            error.limit_refusal(),
            Some(LimitRefusal::Memory {
                bound: LimitBound::PathSearchBudget,
                ..
            })
        ),
        "{error}"
    );
    assert_rows(&db, "MATCH (m:Mark) RETURN count(m)", &[&["0"]]);
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST {search} RETURN length(p)"),
        &[&["11"]],
    );
}

/// Row 22: a long search stops at the query timeout, within about one chunk of work:
/// eighty nodes, all joined, have 6,400 shortest paths between each of their 6,400
/// pairs.
#[test]
fn a_long_search_stops_at_the_deadline() {
    const LIMIT: Duration = Duration::from_millis(250);
    const SLACK: Duration = Duration::from_secs(4);
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(LIMIT)).unwrap();
    db.execute("FOR i IN range(1, 80) INSERT (:Spot {i: i})")
        .unwrap();
    db.execute("MATCH (a:Spot), (b:Spot) INSERT (a)-[:R]->(b)")
        .unwrap();
    let started = Instant::now();
    let error = db
        .execute("MATCH ALL SHORTEST (a:Spot)-[:R]->(b)-[:R]->+(c)-[:R]->(d) RETURN count(*)")
        .expect_err("41 million paths take longer than the timeout");
    let elapsed = started.elapsed();
    assert!(error.to_string().contains("timeout"), "{error}");
    assert!(elapsed < LIMIT + SLACK, "stopped after {elapsed:?}");
    // The database still answers
    assert_rows(&db, "MATCH (s:Spot) RETURN count(s)", &[&["80"]]);
}

// --- 23 and 24: what stays refused -----------------------------------------------------

/// Row 23: a parenthesized subpattern or an alternation inside a selective path pattern
/// is not supported (GRAFEO-Q004), and the message names what is refused.
#[test]
fn a_parenthesized_or_alternated_pattern_is_not_supported() {
    let db = chain();
    for query in [
        "MATCH ANY SHORTEST ((a)-[:R]->(b)){1,2} RETURN count(*)",
        "MATCH ALL SHORTEST (a)-[:R]->(b) | (a)-[:S]->(b) RETURN count(*)",
        "MATCH ANY ((a)-[:R]->(b)-[:S]->(c)){1,2} RETURN count(*)",
    ] {
        let error = db.execute(query).expect_err(query);
        assert_eq!(
            error.error_code(),
            ErrorCode::QueryUnsupported,
            "{query}: {error}"
        );
        let message = error.to_string();
        assert!(
            message.contains("parenthesized") && message.contains("alternated"),
            "{query}: {message}"
        );
    }
}

/// The message of `query`, which is not supported (GRAFEO-Q004).
fn unsupported(db: &GrafeoDB, query: &str) -> String {
    let error = db.execute(query).expect_err(query);
    assert_eq!(
        error.error_code(),
        ErrorCode::QueryUnsupported,
        "{query}: {error}"
    );
    error.to_string()
}

/// Beside row 23: a variable bound at two places of one selective path pattern is not
/// supported, and the message says so (the search would compare two places of one
/// path).
#[test]
fn a_variable_at_two_places_of_the_path_is_not_supported() {
    let db = chain();
    for (query, name) in [
        (
            "MATCH ANY SHORTEST (a)-[:R]->(x)-[:S]->(x)-[:T]->(d) RETURN count(*)",
            "x",
        ),
        (
            "MATCH ALL SHORTEST (a)-[e:R]->(b)-[e:S]->(c) RETURN count(*)",
            "e",
        ),
    ] {
        let message = unsupported(&db, query);
        assert!(
            message.ends_with(&format!(
                "a shortest path search that binds '{name}' at two places of its path pattern \
                 is not supported"
            )),
            "{query}: {message}"
        );
        assert!(!message.contains("parenthesized"), "{query}: {message}");
    }
}

/// Beside row 23: an element pattern whose condition reads another element of the same
/// selective path pattern is not supported, and the message names the element read.
#[test]
fn a_condition_that_reads_another_element_of_the_path_is_not_supported() {
    let db = chain();
    for (query, name) in [
        (
            "MATCH ANY SHORTEST (a)-[:R]->(b WHERE b.name = c.name)-[:S]->(c)-[:T]->(d) \
             RETURN count(*)",
            "c",
        ),
        (
            "MATCH ALL SHORTEST (a)-[:R]->(b)-[s:S WHERE s.k = b.name]->(c) RETURN count(*)",
            "b",
        ),
    ] {
        let message = unsupported(&db, query);
        assert!(
            message.ends_with(&format!(
                "a shortest path search with an element pattern that reads '{name}', another \
                 element of its path pattern, is not supported"
            )),
            "{query}: {message}"
        );
        assert!(!message.contains("parenthesized"), "{query}: {message}");
    }
}

/// A search over a lone node pattern keeps one zero-length path per node: the partition
/// is the pair of the node with itself.
#[test]
fn a_search_over_a_lone_node_keeps_one_path_per_node() {
    let db = chain();
    for prefix in ["ANY SHORTEST", "ALL SHORTEST"] {
        assert_rows(
            &db,
            &format!("MATCH {prefix} (a:Person) RETURN a.name"),
            &[
                &["Alix"],
                &["Django"],
                &["Gus"],
                &["Jules"],
                &["Mia"],
                &["Vincent"],
            ],
        );
    }
}

/// Row 24: openCypher's `shortestPath` takes one relationship pattern; Cypher keeps its
/// refusal of two.
#[cfg(feature = "cypher")]
#[test]
fn cypher_shortest_path_over_two_relationships_stays_refused() {
    let db = chain();
    for query in [
        "MATCH p = shortestPath((a {name: 'Alix'})-[:R]->(b)-[:S*]->(c)) RETURN length(p)",
        "MATCH p = allShortestPaths((a {name: 'Alix'})-[:R]->(b)-[:S]->(c)) RETURN length(p)",
    ] {
        let error = db.execute_cypher(query).expect_err(query);
        assert!(
            error.to_string().contains("exactly one relationship"),
            "{query}: {error}"
        );
    }
}

// --- 11: SQL/PGQ answers as GQL does ---------------------------------------------------

/// SQL/PGQ's `MATCH` is GQL's: a selective pattern over several edge patterns gives the
/// same rows in both.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_answers_as_gql() {
    let db = chain();
    for prefix in [
        "ANY SHORTEST",
        "ALL SHORTEST",
        "SHORTEST 2",
        "ANY",
        "ANY 2",
        "SHORTEST 2 GROUPS",
        "ANY SHORTEST TRAIL",
        "ALL SHORTEST ACYCLIC",
        "SHORTEST 2 SIMPLE GROUPS",
    ] {
        let sql = format!(
            "SELECT * FROM GRAPH_TABLE (MATCH p = {prefix} (a:Person)-[r:R]->(b)-[s:S]->*(c)\
             -[t:T]->(d) COLUMNS (a.name AS a, d.name AS d, PATH_LENGTH(p) AS len, \
             b.name AS b, t.k AS t))"
        );
        let gql = format!(
            "MATCH p = {prefix} (a:Person)-[r:R]->(b)-[s:S]->*(c)-[t:T]->(d) \
             RETURN a.name AS a, d.name AS d, path_length(p) AS len, b.name AS b, t.k AS t"
        );
        let mut from_sql: Vec<String> = db
            .execute_language(&sql, "sql-pgq", None)
            .unwrap_or_else(|error| panic!("{sql}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        from_sql.sort();
        let mut from_gql: Vec<String> = db
            .execute(&gql)
            .unwrap_or_else(|error| panic!("{gql}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        from_gql.sort();
        assert_eq!(from_sql, from_gql, "{prefix}");
        assert!(!from_sql.is_empty(), "{prefix}");
    }
}

// --- 19 again: a quantified edge's variable is its group, on each row -------------------

/// A tree, one path for each pair: Alix -R(1)-> Gus; Gus -S(2)-> Vincent -S(3)-> Jules
/// -T(5)-> Mia; Gus -S(7)-> Butch -T(11)-> Django. Each edge's `w` is in brackets.
fn tree() -> GrafeoDB {
    graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
            ("Django", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "x1", ", w: 1"),
            ("Gus", "S", "Vincent", "s1", ", w: 2"),
            ("Vincent", "S", "Jules", "s2", ", w: 3"),
            ("Jules", "T", "Mia", "t1", ", w: 5"),
            ("Gus", "S", "Butch", "s3", ", w: 7"),
            ("Butch", "T", "Django", "t2", ", w: 11"),
        ],
    )
}

/// Patterns over [`tree`], each with its quantifiers bounded for the unselected match:
/// the quantified edge between fixed ones, first, and beside another quantified one.
const TREE_PATTERNS: [(&str, &str); 3] = [
    (
        "(a)-[x:R]->(b)-[r:S]->+(c)-[t:T]->(d)",
        "(a)-[x:R]->(b)-[r:S]->{1,7}(c)-[t:T]->(d)",
    ),
    (
        "(a)-[r:S]->+(c)-[t:T]->(d)",
        "(a)-[r:S]->{1,7}(c)-[t:T]->(d)",
    ),
    (
        "(a)-[q:R]->*(b)-[r:S]->+(c)-[t:T]->(d)",
        "(a)-[q:R]->{0,7}(b)-[r:S]->{1,7}(c)-[t:T]->(d)",
    ),
];

/// The group variable of a quantified edge pattern among several binds the list of its
/// run, and an aggregate over it reads that list on each row (a horizontal aggregate),
/// as in the unselected match of the same pattern: on the tree every pair has one path,
/// so each selection gives the unselected rows.
#[test]
fn a_group_variable_reads_its_own_run_on_each_row() {
    let db = tree();
    for (pattern, bounded) in TREE_PATTERNS {
        for returned in [
            "a.name, d.name, sum(r.w) AS s",
            "a.name, d.name, avg(r.w) AS s",
            "a.name, d.name, count(r) AS s",
            "a.name, d.name, path_length(r) AS s",
            "a.name, d.name, size(r) AS s",
            "a.name, d.name, [e IN r | e.w] AS s",
            "a.name, sum(r.w) AS s, count(*) AS n",
        ] {
            let unselected = rows(&db, &format!("MATCH {bounded} RETURN {returned}"));
            assert!(!unselected.is_empty(), "{bounded} RETURN {returned}");
            for prefix in ["ANY SHORTEST", "ALL SHORTEST", "ANY"] {
                assert_eq!(
                    rows(&db, &format!("MATCH {prefix} {pattern} RETURN {returned}")),
                    unselected,
                    "{prefix} {pattern} RETURN {returned}"
                );
            }
        }
    }
    // By hand: Alix to Mia over 2 and 3, Alix to Django over 7
    assert_rows(
        &db,
        "MATCH ANY SHORTEST (a {name: 'Alix'})-[:R]->(b)-[r:S]->+(c)-[:T]->(d) \
         RETURN d.name, sum(r.w), size(r), [e IN r | e.w]",
        &[&["Django", "7", "1", "[7]"], &["Mia", "5", "2", "[2, 3]"]],
    );
}

/// SQL/PGQ reads a group variable's path functions as the run's: `PATH_LENGTH(r)` and
/// `EDGES(r)` are those of the run, as in the unselected match; `NODES(r)` of one edge
/// pattern among several is not supported.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_reads_the_path_functions_of_a_group_variable() {
    let db = tree();
    let table = |r#match: &str, columns: &str| {
        let query = format!("SELECT * FROM GRAPH_TABLE (MATCH {match} COLUMNS ({columns}))");
        let mut rows: Vec<String> = db
            .execute_language(&query, "sql-pgq", None)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        rows.sort();
        rows
    };
    for (pattern, bounded) in TREE_PATTERNS {
        for columns in [
            "a.name AS a, d.name AS d, PATH_LENGTH(r) AS len",
            "a.name AS a, d.name AS d, EDGES(r) AS es",
        ] {
            let unselected = table(bounded, columns);
            assert!(!unselected.is_empty(), "{bounded} {columns}");
            assert_eq!(
                table(&format!("ANY SHORTEST {pattern}"), columns),
                unselected,
                "ANY SHORTEST {pattern} {columns}"
            );
        }
    }
    let query = "SELECT * FROM GRAPH_TABLE (MATCH ANY SHORTEST (a)-[x:R]->(b)-[r:S]->+(c) \
                 COLUMNS (NODES(r) AS ns))";
    let error = db
        .execute_language(query, "sql-pgq", None)
        .expect_err(query);
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported, "{error}");
    assert!(
        error.to_string().ends_with(
            "the nodes of 'r', a group variable of one of several edge patterns of a path \
             search, are not supported"
        ),
        "{error}"
    );
}

// --- The plan passes over a search of several edge patterns -------------------------------

/// Each pass that reads or rewrites a search reaches the edge patterns after the first:
/// a filter after the search on a node between them (push-down), a narrow RETURN
/// (projection pruning), the variables a later clause, an OPTIONAL MATCH, a WITH, a
/// subquery and a comma pattern join on (variable collection), a parameter in a node
/// pattern between (parameter substitution), and the binder.
#[test]
fn the_plan_passes_reach_every_edge_pattern() {
    let db = chain();
    let search = "ALL SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)-[:T]->(d)";
    // Filter push-down: the filter reads `c`, which the search binds, so it holds
    // after the selection
    assert_rows(
        &db,
        &format!("MATCH p = {search} WHERE c.name = 'Vincent' RETURN [e IN edges(p) | e.k]"),
        &[&["[r1, s1, t2]"]],
    );
    // A filter on the length after the search: pushed below the selection it would
    // keep the longer path r1 s1 s2 t1
    assert_rows(
        &db,
        &format!("MATCH p = {search} WHERE length(p) = 4 RETURN [e IN edges(p) | e.k]"),
        &[],
    );
    // Projection pruning: a narrow RETURN keeps the search's rows
    assert_rows(&db, &format!("MATCH {search} RETURN count(*)"), &[&["2"]]);
    assert_rows(
        &db,
        &format!("MATCH {search} RETURN c.name"),
        &[&["Jules"], &["Vincent"]],
    );
    // Variable collection: a later MATCH joins on `c`, a WITH passes it on
    assert_rows(
        &db,
        &format!("MATCH {search} MATCH (c)-[:S]->(y) RETURN c.name, y.name"),
        &[&["Vincent", "Jules"]],
    );
    assert_rows(
        &db,
        &format!("MATCH {search} WITH c, d WHERE c.name <> 'Jules' RETURN c.name, d.name"),
        &[&["Vincent", "Mia"]],
    );
    // A comma pattern shares `b` and `c` with the search
    assert_rows(
        &db,
        &format!("MATCH {search}, (b)-[:S]->(c) RETURN c.name"),
        &[&["Jules"], &["Vincent"]],
    );
    // OPTIONAL MATCH: a source without a path keeps its row, with nulls
    assert_rows(
        &db,
        "MATCH (a:Person) WHERE a.name IN ['Alix', 'Django'] \
         OPTIONAL MATCH ALL SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(d) RETURN a.name, c.name",
        &[
            &["Alix", "Jules"],
            &["Alix", "Vincent"],
            &["Django", "null"],
        ],
    );
    // Subqueries
    assert_rows(
        &db,
        "MATCH (a:Person) WHERE a.name IN ['Alix', 'Gus'] RETURN a.name, \
         EXISTS { MATCH ANY SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(d {name: 'Mia'}) } AS e",
        &[&["Alix", "true"], &["Gus", "false"]],
    );
    assert_rows(
        &db,
        "MATCH (a {name: 'Alix'}) CALL { WITH a \
         MATCH ALL SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(d) RETURN count(*) AS n } \
         RETURN a.name, n",
        &[&["Alix", "2"]],
    );
    // Parameter substitution in the node pattern between the edge patterns
    let params = std::collections::HashMap::from([
        ("a".to_string(), Value::String("Alix".into())),
        ("c".to_string(), Value::String("Jules".into())),
    ]);
    let query = "MATCH p = ALL SHORTEST (a {name: $a})-[:R]->(b)-[:S]->*(c {name: $c})-[:T]->(d) \
                 RETURN [e IN edges(p) | e.k]";
    let found: Vec<String> = db
        .execute_with_params(query, params)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| text(&row[0]))
        .collect();
    assert_eq!(found, ["[r1, s3, t1]"], "{query}");
    // The binder: a node between the edge patterns is a node, not an edge
    let query = "MATCH ANY SHORTEST (a)-[:R]->(b)-[:S]->(c) MATCH ()-[b]->() RETURN count(*)";
    let error = db.execute(query).expect_err(query).to_string();
    assert!(error.contains("'b'"), "{query}: {error}");
}

// --- A questioned edge pattern ---------------------------------------------------------

/// A questioned edge pattern (`->?`, zero or one edge) in a selective path pattern is not
/// supported: the search read it as exactly one edge and left out the paths without
/// it, for one edge pattern and for several. The unselected match still reads it as an
/// optional hop.
#[test]
fn a_questioned_edge_in_a_selective_pattern_is_not_supported() {
    let db = chain();
    for (query, what) in [
        (
            "MATCH ANY SHORTEST (a {name: 'Alix'})-[:R]->?(b) RETURN b.name",
            "a shortest",
        ),
        (
            "MATCH ALL SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->?(c) RETURN c.name",
            "a shortest",
        ),
        (
            "MATCH ANY (a)-[:R]->?(b)-[:S]->+(c) RETURN c.name",
            "an ANY",
        ),
    ] {
        let message = unsupported(&db, query);
        assert!(
            message.ends_with(&format!(
                "{what} path search over a questioned edge pattern (`?`) is not supported"
            )),
            "{query}: {message}"
        );
    }
    assert_rows(
        &db,
        "MATCH (a {name: 'Gus'})-[:S]->(b)-[:S]->?(c) RETURN b.name, c.name",
        &[&["Jules", "null"], &["Vincent", "Jules"]],
    );
}

// --- Variables bound before the search ---------------------------------------------------

/// The rows of `query`, which must equal those of `inline`, the same search with the
/// variable bound before written into the pattern instead.
fn assert_same(db: &GrafeoDB, query: &str, inline: &str, want: &[&[&str]]) {
    assert_rows(db, query, want);
    assert_rows(db, inline, want);
}

/// A node between the edge patterns, or a later edge, bound before the search: the
/// search goes on from the rows that bind it (an earlier MATCH, or an earlier part of
/// the same MATCH) and keeps to it, as the same search with it written into the pattern.
#[test]
fn a_variable_bound_before_keeps_the_search_to_it() {
    let db = chain();
    let edges = "RETURN [e IN edges(p) | e.k]";
    assert_same(
        &db,
        &format!(
            "MATCH (m {{name: 'Vincent'}}) MATCH p = ALL SHORTEST (a {{name: 'Alix'}})-[:R]->(b)\
             -[:S]->*(m)-[:T]->(d) {edges}"
        ),
        &format!(
            "MATCH p = ALL SHORTEST (a {{name: 'Alix'}})-[:R]->(b)-[:S]->*(m {{name: 'Vincent'}})\
             -[:T]->(d) {edges}"
        ),
        &[&["[r1, s1, t2]"]],
    );
    assert_same(
        &db,
        &format!(
            "MATCH (m {{name: 'Jules'}}), p = ALL SHORTEST (a {{name: 'Alix'}})-[:R]->(b)\
             -[:S]->*(m)-[:T]->(d) {edges}"
        ),
        &format!(
            "MATCH p = ALL SHORTEST (a {{name: 'Alix'}})-[:R]->(b)-[:S]->*(m {{name: 'Jules'}})\
             -[:T]->(d) {edges}"
        ),
        &[&["[r1, s3, t1]"]],
    );
    assert_same(
        &db,
        &format!(
            "MATCH ()-[e:S {{k: 's2'}}]->() MATCH p = ALL SHORTEST (a {{name: 'Alix'}})-[:R]->(b)\
             -[:S]->(c)-[e]->(f)-[:T]->(d) {edges}"
        ),
        &format!(
            "MATCH p = ALL SHORTEST (a {{name: 'Alix'}})-[:R]->(b)-[:S]->(c)-[e:S {{k: 's2'}}]->(f)\
             -[:T]->(d) {edges}"
        ),
        &[&["[r1, s1, s2, t1]"]],
    );
}

/// An endpoint's name inside the path is that endpoint: the source again (the path comes
/// back to it) or the target (the path passes it before it ends there), and under ANY the
/// search then pairs the source with the target instead of with every node. A condition
/// between that reads the target does the same.
#[test]
fn an_endpoint_named_inside_the_path_is_that_endpoint() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "S", "Alix", "s1", ""),
            ("Gus", "S", "Vincent", "s2", ""),
            ("Vincent", "T", "Gus", "t1", ""),
            ("Alix", "T", "Mia", "t2", ""),
            ("Vincent", "T", "Mia", "t3", ""),
        ],
    );
    for selection in SELECTIONS {
        check(
            &db,
            selection,
            "",
            "(a)-[:R]->(b)-[:S]->(a)-[:T]->(d)",
            "(a)-[:R]->(b)-[:S]->(a)-[:T]->(d)",
        );
        check(
            &db,
            selection,
            "",
            "(a)-[:R]->(d)-[:S]->(c)-[:T]->(d)",
            "(a)-[:R]->(d)-[:S]->(c)-[:T]->(d)",
        );
        check(
            &db,
            selection,
            "",
            "(a)-[:R]->(b WHERE b.name <> d.name)-[:S]->*(c)-[:T]->(d)",
            "(a)-[:R]->(b)-[:S]->{0,4}(c)-[:T]->(d) WHERE b.name <> d.name",
        );
    }
    assert_rows(
        &db,
        "MATCH p = ANY (a)-[:R]->(d)-[:S]->(c)-[:T]->(d) RETURN a.name, d.name, \
         [e IN edges(p) | e.k]",
        &[&["Alix", "Gus", "[r1, s2, t1]"]],
    );
}

// --- More of the edge-case matrix ----------------------------------------------------------

/// A whole path of length 0: every quantifier takes none, and the node pairs with itself.
#[test]
fn a_whole_path_of_length_zero_pairs_a_node_with_itself() {
    let db = chain();
    assert_rows(
        &db,
        "MATCH p = ANY SHORTEST (a {name: 'Django'})-[x:R]->*(b)-[y:S]->*(c) \
         RETURN c.name, length(p), size(nodes(p)), size(x), size(y)",
        &[&["Django", "0", "1", "0", "0"]],
    );
    check_all(
        &db,
        "(a)-[:R]->*(b)-[:S]->*(c)",
        "(a)-[:R]->{0,6}(b)-[:S]->{0,6}(c)",
    );
}

/// The source and the target as one variable: the paths that come back to their start.
#[test]
fn source_and_target_as_one_variable() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "S", "Vincent", "s1", ""),
            ("Vincent", "T", "Alix", "t1", ""),
            ("Gus", "S", "Jules", "s2", ""),
            ("Jules", "S", "Vincent", "s3", ""),
            ("Jules", "T", "Alix", "t2", ""),
        ],
    );
    check_all(
        &db,
        "(a)-[:R]->(b)-[:S]->*(c)-[:T]->(a)",
        "(a)-[:R]->(b)-[:S]->{0,4}(c)-[:T]->(a)",
    );
    assert_rows(
        &db,
        "MATCH p = ALL SHORTEST (a)-[:R]->(b)-[:S]->*(c)-[:T]->(a) RETURN a.name, \
         [e IN edges(p) | e.k]",
        &[&["Alix", "[r1, s1, t1]"], &["Alix", "[r1, s2, t2]"]],
    );
}

/// LIMIT after a search keeps its first rows, in a given order.
#[test]
fn a_limit_after_the_search() {
    let db = chain();
    let search = "MATCH p = ALL SHORTEST (a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)-[:T]->(d)";
    assert_rows(
        &db,
        &format!("{search} RETURN [e IN edges(p) | e.k] AS ks ORDER BY ks LIMIT 1"),
        &[&["[r1, s1, t2]"]],
    );
    assert_rows(
        &db,
        &format!("{search} WITH p LIMIT 1 RETURN count(*)"),
        &[&["1"]],
    );
}

/// A search whose rows fill several output chunks (a small budget ends a chunk after a
/// few ids) returns every row, as the unselected match does.
#[test]
fn rows_stream_across_output_chunks() {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(16 * 1024)).unwrap();
    db.execute("UNWIND range(1, 100) AS i INSERT (:P {i: i})-[:R]->(:Q {i: i})-[:S]->(:T {i: i})")
        .unwrap();
    let unselected = rows(&db, "MATCH (a:P)-[:R]->(b)-[:S]->(c) RETURN a.i, c.i");
    assert_eq!(unselected.len(), 100);
    for prefix in ["ANY", "ANY SHORTEST", "ALL SHORTEST"] {
        assert_eq!(
            rows(
                &db,
                &format!("MATCH {prefix} (a:P)-[:R]->(b)-[:S]->(c) RETURN a.i, c.i")
            ),
            unselected,
            "{prefix}"
        );
    }
}

/// A group variable returned as it is: the list of its run's edges, as in the unselected
/// match.
#[test]
fn a_group_variable_returned_raw_is_its_edges() {
    let db = tree();
    let raw = |query: &str| -> Vec<String> {
        let mut rows: Vec<String> = db
            .execute(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        rows.sort();
        rows
    };
    for (pattern, bounded) in TREE_PATTERNS {
        let unselected = raw(&format!("MATCH {bounded} RETURN a.name, d.name, r"));
        assert!(!unselected.is_empty(), "{bounded}");
        assert_eq!(
            raw(&format!(
                "MATCH ANY SHORTEST {pattern} RETURN a.name, d.name, r"
            )),
            unselected,
            "{pattern}"
        );
    }
}

/// GQL's `shortestPath` and `allShortestPaths` over a chain of edge patterns run as ANY
/// SHORTEST and ALL SHORTEST (openCypher's take one relationship pattern, and Cypher
/// keeps that refusal).
#[test]
fn gql_shortest_path_functions_run_over_a_chain() {
    let db = chain();
    let pattern = "(a {name: 'Alix'})-[:R]->(b)-[:S]->*(c)-[:T]->(d)";
    let returned = "RETURN d.name, length(p), [e IN edges(p) | e.k]";
    assert_eq!(
        rows(
            &db,
            &format!("MATCH p = shortestPath({pattern}) {returned}")
        ),
        rows(&db, &format!("MATCH p = ANY SHORTEST {pattern} {returned}")),
    );
    assert_eq!(
        rows(
            &db,
            &format!("MATCH p = allShortestPaths({pattern}) {returned}")
        ),
        rows(&db, &format!("MATCH p = ALL SHORTEST {pattern} {returned}")),
    );
    assert_eq!(
        rows(
            &db,
            &format!("MATCH p = allShortestPaths({pattern}) {returned}")
        )
        .len(),
        2
    );
}

/// `shortestPath` over a lone node pattern names what it lacks.
#[test]
fn a_shortest_path_function_over_a_lone_node_says_so() {
    let db = chain();
    let message = unsupported(&db, "MATCH p = shortestPath((a {name: 'Alix'})) RETURN p");
    assert!(
        message.ends_with("a shortest path search over a lone node pattern is not supported"),
        "{message}"
    );
}

/// A chain of more edge patterns than a search can count is refused before it runs.
#[test]
fn a_chain_of_too_many_edge_patterns_is_refused() {
    let db = chain();
    let chain = "-[:R]->()".repeat(usize::from(u16::MAX) + 1);
    let query = format!("MATCH ANY SHORTEST (a){chain} RETURN count(*)");
    let message = unsupported(&db, &query);
    assert!(
        message.ends_with(
            "a shortest path search over more than 65535 edge patterns is not supported"
        ),
        "{}",
        &message[message.len().saturating_sub(300)..]
    );
}

/// EXPLAIN shows every edge pattern of a search, and the node patterns between.
#[test]
fn explain_shows_every_edge_pattern() {
    let db = chain();
    let result = db
        .execute(
            "EXPLAIN MATCH ANY SHORTEST (a)-[:R]->(b:Person)<-[r:S]-{1,3}(c)-[:T]-(d) RETURN d",
        )
        .unwrap();
    let text = match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    };
    assert!(
        text.contains("[path: -[:R]->(b)<-[r:S]-{1,3}(c)-[:T]-]"),
        "{text}"
    );
    assert!(text.contains("[node: hasLabel"), "{text}");
}

// --- The memory a search over several edge patterns holds ---------------------------------

/// Nodes 0 to 8 in a row, 100 parallel `R` edges between each two, then an `S` edge:
/// 100^8 shortest walks over the `R` edges.
fn wide() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(16 * 1024)).unwrap();
    db.execute("FOR i IN range(0, 8) INSERT (:Stop {i: i})")
        .unwrap();
    db.execute(
        "MATCH (s:Stop), (t:Stop) WHERE t.i = s.i + 1 FOR k IN range(1, 100) INSERT (s)-[:R]->(t)",
    )
    .unwrap();
    db.execute("MATCH (s:Stop {i: 8}) INSERT (s)-[:S]->(:End)")
        .unwrap();
    db
}

/// The predecessors a WALK search over states records count against the path search
/// budget: ALL SHORTEST, which keeps every one, is refused with GRAFEO-S001, while ANY
/// SHORTEST keeps one per state and answers under the same budget.
#[test]
fn a_walk_search_holds_its_predecessors_within_the_budget() {
    let db = wide();
    let search = "(a:Stop {i: 0})-[:R]->+(b)-[:S]->(c:End)";
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST {search} RETURN length(p)"),
        &[&["9"]],
    );
    let error = db
        .execute(&format!("MATCH p = ALL SHORTEST {search} RETURN count(p)"))
        .expect_err("100^8 walks and 800 predecessors do not fit in 16 KiB");
    assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
    assert!(
        matches!(
            error.limit_refusal(),
            Some(LimitRefusal::Memory {
                bound: LimitBound::PathSearchBudget,
                ..
            })
        ),
        "{error}"
    );
}

/// The frontier and the predecessors are charged before the walks are read back: a
/// search with no end in reach still stops at the budget instead of growing.
#[test]
fn a_walk_search_without_an_end_stops_at_the_budget() {
    let db = wide();
    let error = db
        .execute(
            "MATCH SHORTEST 3 GROUPS (a:Stop {i: 0})-[:R]->+(b)-[:S]->(c:Stop) RETURN count(*)",
        )
        .expect_err("the predecessors of 8 layers of 100 edges pass 16 KiB");
    assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
}

/// SHORTEST 3 over the wide rungs keeps three walks of the shortest length, each with
/// its own edges: a state keeps the first three steps into it.
#[test]
fn shortest_three_over_wide_layers_keeps_three_distinct_walks() {
    let db = wide();
    let found = rows(
        &db,
        "MATCH p = SHORTEST 3 (a:Stop {i: 0})-[:R]->+(b)-[:S]->(c:End) \
         RETURN length(p), [e IN edges(p) | id(e)]",
    );
    assert_eq!(found.len(), 3, "{found:?}");
    assert!(found.iter().all(|row| row[0] == "9"), "{found:?}");
    let mut edges: Vec<&String> = found.iter().map(|row| &row[1]).collect();
    edges.dedup();
    assert_eq!(edges.len(), 3, "three distinct walks: {found:?}");
}

/// ALL SHORTEST keeps every step into a state, and those steps count against the budget
/// before a walk is read back: 2,000 parallel edges into one node pass 64 KiB while the
/// search records them, where ANY SHORTEST keeps one and answers.
#[test]
fn all_shortest_is_refused_for_its_steps_alone() {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(64 * 1024)).unwrap();
    db.execute("INSERT (:Hub)-[:S]->(:Tip)").unwrap();
    db.execute("INSERT (:Start)").unwrap();
    db.execute("MATCH (s:Start), (h:Hub) FOR k IN range(1, 2000) INSERT (s)-[:R]->(h)")
        .unwrap();
    let search = "(a:Start)-[:R]->+(b)-[:S]->(c:Tip)";
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST {search} RETURN length(p)"),
        &[&["2"]],
    );
    let error = db
        .execute(&format!("MATCH p = ALL SHORTEST {search} RETURN count(p)"))
        .expect_err("2,000 steps into one node pass 64 KiB");
    assert_eq!(error.error_code(), ErrorCode::StorageFull, "{error}");
    // Refused for the memory it holds, not for the paths it keeps
    let message = error.to_string();
    assert!(
        message.contains("would hold more than the 64 KiB of memory a path search may use")
            && !message.contains("at once"),
        "refused while recording the steps, not reading walks back: {error}"
    );
}

/// SQL/PGQ refuses a questioned edge pattern in a selective pattern as GQL does.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_refuses_a_questioned_edge_in_a_selective_pattern() {
    let db = chain();
    let query = "SELECT * FROM GRAPH_TABLE (MATCH ANY SHORTEST (a WHERE a.name = 'Alix')\
                 -[:R]->?(b) COLUMNS (b.name AS b))";
    let error = db
        .execute_language(query, "sql-pgq", None)
        .expect_err(query);
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported, "{error}");
    assert!(
        error.to_string().ends_with(
            "a shortest path search over a questioned edge pattern (`?`) is not supported"
        ),
        "{error}"
    );
}

/// Past its minimum, a state is only dominated by states with as many edges of its edge
/// pattern or fewer: here the walk that reaches the middle node later, with fewer `R`
/// edges taken, is the one that can still take the two more the end needs. Three `Q`
/// edges then one `R` reach it at depth 4 with one hop; three `R` edges reach it at depth
/// 3 with three, of the four `{1,4}` allows.
#[test]
fn a_state_with_more_edges_left_is_not_dominated() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
            ("Django", ":Person"),
            ("Hans", ":Person"),
            ("Beatrix", ":Person"),
            ("Shosanna", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Vincent", "R", "Jules", "r3", ""),
            ("Alix", "Q", "Mia", "q1", ""),
            ("Mia", "Q", "Butch", "q2", ""),
            ("Butch", "Q", "Django", "q3", ""),
            ("Django", "R", "Jules", "r4", ""),
            ("Jules", "R", "Hans", "r5", ""),
            ("Hans", "R", "Beatrix", "r6", ""),
            ("Beatrix", "S", "Shosanna", "s1", ""),
        ],
    );
    let pattern = "(a {name: 'Alix'})-[:Q]->*(b)-[:R]->{1,4}(c)-[:S]->(d {name: 'Shosanna'})";
    let bounded = "(a {name: 'Alix'})-[:Q]->{0,10}(b)-[:R]->{1,4}(c)-[:S]->(d {name: 'Shosanna'})";
    for selection in SELECTIONS {
        check(&db, selection, "", pattern, bounded);
    }
    assert_rows(
        &db,
        &format!("MATCH p = ANY SHORTEST {pattern} RETURN [e IN edges(p) | e.k]"),
        &[&["[q1, q2, q3, r4, r5, r6, s1]"]],
    );
}

/// The search with bounded quantifiers against the bounded oracle, on small graphs drawn
/// from a fixed seed with cycles and parallel edges, for random bounds and every
/// selection: the states it leaves out past a minimum change no answer.
#[test]
fn bounded_quantifiers_agree_with_the_oracle_on_drawn_graphs() {
    let names = [
        "Alix", "Gus", "Vincent", "Jules", "Mia", "Butch", "Django", "Hans",
    ];
    let mut state: u64 = 0x0788_D0A1;
    let mut draw = |below: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % below
    };
    for _ in 0..8 {
        let size = 4 + usize::try_from(draw(4)).unwrap();
        let nodes: Vec<Node> = names[..size]
            .iter()
            .map(|name| (*name, ":Person"))
            .collect();
        let pick = |drawn: u64| names[usize::try_from(drawn).unwrap()];
        let width = u64::try_from(size).unwrap();
        let mut drawn = Vec::new();
        for index in 0..(size + 3) {
            let from = pick(draw(width));
            let to = pick(draw(width));
            let kind = if draw(3) == 0 { "S" } else { "R" };
            drawn.push((from, kind, to, format!("e{index}")));
        }
        let edges: Vec<Edge> = drawn
            .iter()
            .map(|(from, kind, to, key)| (*from, *kind, *to, key.as_str(), ""))
            .collect();
        let db = graph(&nodes, &edges);
        let min = draw(3);
        let max = min + 1 + draw(3);
        let min2 = draw(2);
        let max2 = min2 + 1 + draw(2);
        // A bounded run then a fixed edge, two bounded runs of one type, and a fixed
        // edge then a bounded run; `check` names the pattern in a failure
        for pattern in [
            format!("(a)-[:R]->{{{min},{max}}}(b)-[:S]->(c)"),
            format!("(a)-[:R]->{{{min},{max}}}(b)-[:R]->{{{min2},{max2}}}(c)"),
            format!("(a)-[:S]->(b)-[:R]->{{{min},{max}}}(c)"),
        ] {
            for selection in SELECTIONS {
                check(&db, selection, "", &pattern, &pattern);
            }
        }
    }
}

/// A bounded quantifier with a large maximum answers like an unbounded one: past its
/// minimum, a state with fewer hops reached earlier stands for the states with more.
#[test]
fn a_large_bounded_quantifier_answers_like_an_unbounded_one() {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_expansion_limit(2_000)).unwrap();
    fill(
        &db,
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Vincent", "R", "Alix", "r3", ""),
            ("Vincent", "S", "Mia", "s1", ""),
        ],
    );
    let returned = "RETURN a.name, c.name, length(p)";
    for selection in SELECTIONS {
        let prefix = selection.prefix();
        assert_eq!(
            rows(
                &db,
                &format!("MATCH p = {prefix} (a)-[:R]->{{1,100000}}(f)-[:S]->(c) {returned}")
            ),
            rows(
                &db,
                &format!("MATCH p = {prefix} (a)-[:R]->+(f)-[:S]->(c) {returned}")
            ),
            "{prefix}"
        );
    }
}

/// Two walks of the first run reach Vincent, with two and with three edges, so the
/// second run's states there come at several depths: an earlier state stands for a
/// later one only at a shorter depth, and only once per distinct depth, as many as the
/// selection keeps lengths. Every selection agrees with the oracle.
#[test]
fn a_dominated_state_counts_distinct_earlier_depths() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
            ("Butch", ":Person"),
            ("Django", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Alix", "R", "Mia", "r3", ""),
            ("Mia", "R", "Butch", "r4", ""),
            ("Butch", "R", "Vincent", "r5", ""),
            ("Vincent", "R", "Jules", "r6", ""),
            ("Jules", "S", "Django", "s1", ""),
        ],
    );
    let pattern = "(a)-[:R]->{0,9}(b)-[:R]->{1,9}(c)-[:S]->(d)";
    for selection in SELECTIONS {
        check(&db, selection, "", pattern, pattern);
    }
}

/// A walk that reaches a node again within the minimum of its run is not dominated by
/// its first visit there, which had fewer edges of the run: from Alix, three `R` edges
/// at the least reach Jules only around the cycle between Gus and Vincent.
#[test]
fn a_state_within_its_minimum_is_not_dominated() {
    let db = graph(
        &[
            ("Alix", ":Person"),
            ("Gus", ":Person"),
            ("Vincent", ":Person"),
            ("Jules", ":Person"),
            ("Mia", ":Person"),
        ],
        &[
            ("Alix", "R", "Gus", "r1", ""),
            ("Gus", "R", "Vincent", "r2", ""),
            ("Vincent", "R", "Gus", "r3", ""),
            ("Gus", "R", "Jules", "r4", ""),
            ("Jules", "S", "Mia", "s1", ""),
        ],
    );
    let pattern = "(a)-[:R]->{3,5}(b)-[:S]->(c)";
    for selection in SELECTIONS {
        check(&db, selection, "", pattern, pattern);
    }
    assert_rows(
        &db,
        "MATCH p = ANY SHORTEST (a {name: 'Alix'})-[:R]->{3,5}(b)-[:S]->(c {name: 'Mia'})          RETURN [e IN edges(p) | e.k]",
        &[&["[r1, r2, r3, r4, s1]"]],
    );
}
