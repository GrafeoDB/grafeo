//! Snapshot isolation of values, labels and deletes, end to end (#412).
//!
//! The guarantees, numbered as in #412:
//!
//! - G1, G2, G3: no other reader sees an open transaction's values, labels
//!   or deletes;
//! - G4: a snapshot reads the same after another session commits;
//! - G5: index reads match the snapshot;
//! - G6: indexes never expose an open transaction's values;
//! - G7: a rollback, and a rollback to a savepoint, restore every path;
//! - G8: a transaction reads its own writes on every path;
//! - G9: no increment of an aborted transaction is kept;
//! - G10: a commit that never publishes stays invisible;
//! - G12: a delete takes effect at its commit epoch, also after a replay.
//!
//! G11 (checkpoints and copies hold the committed state) is
//! `checkpoint_committed_state.rs`. Each guarantee is read through the paths
//! that can break it:
//!
//! - GQL and Cypher queries: a single property (`n.age`, the single-property
//!   read path), filters, pattern keys, `IN` lists and ranges (the property
//!   index with one), an aggregate, the whole node or edge, `properties`,
//!   Cypher's map projection, label scans, label checks, the label union,
//!   `labels`, expands, counts, `text_match`, top-k by similarity and the
//!   search procedures;
//! - the direct API: `get_node`, `get_edge`, `get_node_edges`,
//!   `find_nodes_by_property`, `get_node_at_epoch`, `get_edge_at_epoch`,
//!   `get_node_labels`, the counts, `text_search` and `vector_search`;
//! - read by another session, by another transaction, at an epoch
//!   (`execute_at_epoch`, `get_node_at_epoch`), by the database's own reads
//!   and by the writing transaction itself;
//! - with and without the index that serves the path: a property index on
//!   `age`, a text index on `bio`, a vector index on `emb`.
//!
//! Each test names every read `scenario / change / language / reader /
//! index / path` and checks it with exact rows; a test fails with the list
//! of every read that did not hold. The row-group store (since the H2d
//! switch) keeps every guarantee on every path: there are no known gaps.
//! Nothing here is ignored.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test snapshot_isolation
//! cargo test -p grafeo-engine --features full --test snapshot_isolation
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Direction, GrafeoDB, Session};

/// What the reads of one test found.
#[derive(Default)]
struct Report {
    reads: Vec<(String, std::result::Result<(), String>)>,
}

impl Report {
    fn push(&mut self, name: String, outcome: std::result::Result<(), String>) {
        self.reads.push((name, outcome));
    }

    #[cfg(feature = "testing-statement-injection")]
    fn extend(&mut self, other: Self) {
        self.reads.extend(other.reads);
    }

    /// Checks that every read passed.
    ///
    /// # Panics
    ///
    /// When no read ran, a name repeats, or a read fails.
    fn finish(self) {
        assert!(!self.reads.is_empty(), "no read ran");
        let mut names = HashSet::new();
        let mut problems = String::new();
        for (name, outcome) in &self.reads {
            assert!(names.insert(name.as_str()), "{name} runs twice");
            if let Err(failure) = outcome {
                let _ = writeln!(problems, "{name}: {failure}");
            }
        }
        assert!(problems.is_empty(), "\n{problems}");
    }
}

// ── The graph ───────────────────────────────────────────────────────

/// `(:Person {name: 'Alix', age: 30, bio: 'lives in Prague', emb: [1, 0,
/// 0]})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus', age: 40, bio: 'lives
/// in Berlin', emb: [0.6, 0.8, 0]})`, committed.
#[derive(Clone)]
struct Graph {
    db: Arc<GrafeoDB>,
    alix: NodeId,
    gus: NodeId,
    knows: EdgeId,
}

fn vector(values: [f32; 3]) -> Value {
    Value::Vector(values.to_vec().into())
}

/// The people in a new in-memory database, with `index`.
fn people(index: Index) -> Graph {
    people_in(GrafeoDB::new_in_memory(), index)
}

fn people_in(db: GrafeoDB, index: Index) -> Graph {
    let (alix, gus, knows) = fill(&db, index);
    Graph {
        db: Arc::new(db),
        alix,
        gus,
        knows,
    }
}

/// Writes the people into `db`, with `index`; returns Alix, Gus and their
/// edge.
fn fill(db: &GrafeoDB, index: Index) -> (NodeId, NodeId, EdgeId) {
    let alix = db
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::from("Alix")),
                ("age", Value::Int64(30)),
                ("bio", Value::from("lives in Prague")),
                ("emb", vector([1.0, 0.0, 0.0])),
            ],
        )
        .unwrap();
    let gus = db
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::from("Gus")),
                ("age", Value::Int64(40)),
                ("bio", Value::from("lives in Berlin")),
                ("emb", vector([0.6, 0.8, 0.0])),
            ],
        )
        .unwrap();
    let knows = db
        .create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(2020))])
        .unwrap();
    index.create(db);
    (alix, gus, knows)
}

/// The index a variant runs with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Index {
    None,
    /// A property index on `age`.
    Property,
    /// A text index on `:Person(bio)`.
    #[cfg(feature = "text-index")]
    Text,
    /// A cosine vector index on `:Person(emb)`.
    #[cfg(feature = "vector-index")]
    Vector,
}

impl Index {
    fn name(self) -> &'static str {
        match self {
            Self::None => "no index",
            Self::Property => "property index",
            #[cfg(feature = "text-index")]
            Self::Text => "text index",
            #[cfg(feature = "vector-index")]
            Self::Vector => "vector index",
        }
    }

    fn create(self, db: &GrafeoDB) {
        match self {
            Self::None => {}
            Self::Property => db.create_property_index("age").unwrap(),
            #[cfg(feature = "text-index")]
            Self::Text => db.create_text_index("Person", "bio").unwrap(),
            #[cfg(feature = "vector-index")]
            Self::Vector => db
                .create_vector_index("Person", "emb", Some(3), Some("cosine"), None, None, None)
                .unwrap(),
        }
    }
}

/// The language a variant writes and reads in; `Api` writes and reads
/// through the direct API.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lang {
    Gql,
    #[cfg(feature = "cypher")]
    Cypher,
    Api,
}

impl Lang {
    fn name(self) -> &'static str {
        match self {
            Self::Gql => "gql",
            #[cfg(feature = "cypher")]
            Self::Cypher => "cypher",
            Self::Api => "api",
        }
    }
}

fn languages() -> Vec<Lang> {
    let mut languages = vec![Lang::Gql];
    #[cfg(feature = "cypher")]
    languages.push(Lang::Cypher);
    languages.push(Lang::Api);
    languages
}

fn query_languages() -> Vec<Lang> {
    languages()
        .into_iter()
        .filter(|lang| *lang != Lang::Api)
        .collect()
}

/// Runs `query` in `session`, with `params`.
fn execute(
    session: &Session,
    lang: Lang,
    query: &str,
    params: HashMap<String, Value>,
) -> Result<QueryResult> {
    match lang {
        Lang::Gql => session.execute_with_params(query, params),
        #[cfg(feature = "cypher")]
        Lang::Cypher => session.execute_cypher_with_params(query, params),
        Lang::Api => unreachable!("the API runs no queries"),
    }
}

// ── The changes ─────────────────────────────────────────────────────

/// The write under test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Change {
    /// Alix's age from 30 to 99.
    NodeValue,
    /// The KNOWS edge's `since` from 2020 to 2099.
    EdgeValue,
    /// Alix gets `Admin` and loses `Person`.
    Label,
    /// `DETACH DELETE` of Gus.
    NodeDelete,
    /// `DELETE` of the KNOWS edge.
    EdgeDelete,
    /// Alix's bio from 'lives in Prague' to 'lives in Paris'.
    #[cfg(feature = "text-index")]
    Text,
    /// Alix's embedding from [1, 0, 0] to [0, 0, 1].
    #[cfg(feature = "vector-index")]
    Vector,
}

impl Change {
    fn name(self) -> &'static str {
        match self {
            Self::NodeValue => "node value",
            Self::EdgeValue => "edge value",
            Self::Label => "labels",
            Self::NodeDelete => "node delete",
            Self::EdgeDelete => "edge delete",
            #[cfg(feature = "text-index")]
            Self::Text => "text value",
            #[cfg(feature = "vector-index")]
            Self::Vector => "vector value",
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::NodeValue,
            Self::EdgeValue,
            Self::Label,
            Self::NodeDelete,
            Self::EdgeDelete,
            #[cfg(feature = "text-index")]
            Self::Text,
            #[cfg(feature = "vector-index")]
            Self::Vector,
        ]
    }

    fn is_delete(self) -> bool {
        matches!(self, Self::NodeDelete | Self::EdgeDelete)
    }

    /// Whether the change is read through a search index only.
    fn is_search(self) -> bool {
        match self {
            #[cfg(feature = "text-index")]
            Self::Text => true,
            #[cfg(feature = "vector-index")]
            Self::Vector => true,
            _ => false,
        }
    }

    /// The indexes the change's reads run with.
    fn indexes(self) -> Vec<Index> {
        match self {
            Self::NodeValue | Self::Label => vec![Index::None, Index::Property],
            Self::EdgeValue | Self::EdgeDelete => vec![Index::None],
            Self::NodeDelete => vec![
                Index::None,
                Index::Property,
                #[cfg(feature = "text-index")]
                Index::Text,
                #[cfg(feature = "vector-index")]
                Index::Vector,
            ],
            #[cfg(feature = "text-index")]
            Self::Text => vec![Index::Text],
            #[cfg(feature = "vector-index")]
            Self::Vector => vec![Index::None, Index::Vector],
        }
    }

    /// The statements of the change, in GQL and Cypher alike.
    fn statements(self) -> &'static [&'static str] {
        match self {
            Self::NodeValue => &["MATCH (n:Person {name: 'Alix'}) SET n.age = 99"],
            Self::EdgeValue => &["MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() SET r.since = 2099"],
            Self::Label => &["MATCH (n:Person {name: 'Alix'}) SET n:Admin REMOVE n:Person"],
            Self::NodeDelete => &["MATCH (n:Person {name: 'Gus'}) DETACH DELETE n"],
            Self::EdgeDelete => &["MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() DELETE r"],
            #[cfg(feature = "text-index")]
            Self::Text => &["MATCH (n:Person {name: 'Alix'}) SET n.bio = 'lives in Paris'"],
            #[cfg(feature = "vector-index")]
            Self::Vector => &["MATCH (n:Person {name: 'Alix'}) SET n.emb = $emb"],
        }
    }

    /// Writes the change through `session`, in its transaction if it has
    /// one, else committed at once.
    fn write(self, session: &Session, graph: &Graph, lang: Lang) {
        if lang == Lang::Api {
            self.write_directly(session, graph);
            return;
        }
        for statement in self.statements() {
            let params = HashMap::from([("emb".to_string(), vector([0.0, 0.0, 1.0]))]);
            execute(session, lang, statement, params)
                .unwrap_or_else(|error| panic!("{statement}: {error}"));
        }
    }

    fn write_directly(self, session: &Session, graph: &Graph) {
        match self {
            Self::NodeValue => session
                .set_node_property(graph.alix, "age", Value::Int64(99))
                .unwrap(),
            Self::EdgeValue => session
                .set_edge_property(graph.knows, "since", Value::Int64(2099))
                .unwrap(),
            Self::Label => {
                assert!(session.add_node_label(graph.alix, "Admin").unwrap());
                assert!(session.remove_node_label(graph.alix, "Person").unwrap());
            }
            Self::NodeDelete => assert!(session.detach_delete_node(graph.gus).unwrap()),
            Self::EdgeDelete => assert!(session.delete_edge(graph.knows).unwrap()),
            #[cfg(feature = "text-index")]
            Self::Text => session
                .set_node_property(graph.alix, "bio", Value::from("lives in Paris"))
                .unwrap(),
            #[cfg(feature = "vector-index")]
            Self::Vector => session
                .set_node_property(graph.alix, "emb", vector([0.0, 0.0, 1.0]))
                .unwrap(),
        }
    }
}

// ── Reading ─────────────────────────────────────────────────────────

type Rows = Vec<Vec<Value>>;

/// Who reads.
#[derive(Clone, Copy)]
enum Reader<'a> {
    /// A session, in a transaction or not.
    Session(&'a Session),
    /// A session outside a transaction, at an epoch: `execute_at_epoch`, and
    /// `get_node_at_epoch` and `get_edge_at_epoch` for the direct reads.
    At(&'a Session, EpochId),
    /// The database's own reads (`GrafeoDB::execute`, `GrafeoDB::get_node`).
    Database,
}

/// What a query's rows are reduced to.
#[derive(Clone, Copy)]
enum Pick {
    /// The rows as they are, lists sorted.
    Rows,
    /// The value of a key of the record or map in the first column.
    Key(&'static str),
    /// The labels of the node record in the first column.
    Labels,
    /// The first column (the node of a search procedure's row).
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    First,
}

/// A read through the direct API: `None` where the reader has no such read.
type ApiRead = Arc<dyn Fn(&Graph, Reader<'_>) -> Option<Rows> + Send + Sync>;

#[derive(Clone)]
enum Read {
    /// A query, the same in GQL and Cypher.
    Query(String, Pick),
    /// A query in Cypher only.
    #[cfg(feature = "cypher")]
    Cypher(String, Pick),
    /// A read through the direct API.
    Api(ApiRead),
}

/// One path of a change, and what it reads before and after the change.
#[derive(Clone)]
struct Path {
    name: &'static str,
    read: Read,
    before: Rows,
    after: Rows,
}

fn path(name: &'static str, read: Read, before: Rows, after: Rows) -> Path {
    Path {
        name,
        read,
        before,
        after,
    }
}

fn query(text: &str, pick: Pick) -> Read {
    Read::Query(text.to_string(), pick)
}

fn api(read: impl Fn(&Graph, Reader<'_>) -> Option<Rows> + Send + Sync + 'static) -> Read {
    Read::Api(Arc::new(read))
}

fn id(node: NodeId) -> Value {
    Value::Int64(i64::try_from(node.as_u64()).unwrap())
}

fn edge_id(edge: EdgeId) -> Value {
    Value::Int64(i64::try_from(edge.as_u64()).unwrap())
}

fn ints(values: &[i64]) -> Rows {
    values
        .iter()
        .map(|value| vec![Value::Int64(*value)])
        .collect()
}

fn names(values: &[&str]) -> Rows {
    values
        .iter()
        .map(|value| vec![Value::from(*value)])
        .collect()
}

fn nodes(ids: &[NodeId]) -> Rows {
    ids.iter().map(|node| vec![id(*node)]).collect()
}

fn label_list(labels: &[&str]) -> Rows {
    vec![vec![sorted_list(Value::List(
        labels
            .iter()
            .map(|label| Value::from(*label))
            .collect::<Vec<_>>()
            .into(),
    ))]]
}

fn none() -> Rows {
    Vec::new()
}

fn sorted_list(value: Value) -> Value {
    match value {
        Value::List(items) => {
            let mut items = items.to_vec();
            items.sort_by_key(|item| format!("{item:?}"));
            Value::List(items.into())
        }
        other => other,
    }
}

fn entry(value: &Value, key: &str) -> Value {
    match value {
        Value::Map(map) => map
            .get(&PropertyKey::new(key))
            .cloned()
            .unwrap_or(Value::Null),
        other => panic!("expected a record, read {other:?}"),
    }
}

fn sorted(mut rows: Rows) -> Rows {
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn reduce(rows: &[Vec<Value>], pick: Pick) -> Rows {
    sorted(
        rows.iter()
            .map(|row| match pick {
                Pick::Rows => row.iter().cloned().map(sorted_list).collect(),
                Pick::Key(key) => vec![entry(&row[0], key)],
                Pick::Labels => vec![sorted_list(entry(&row[0], "_labels"))],
                #[cfg(any(feature = "text-index", feature = "vector-index"))]
                Pick::First => vec![row[0].clone()],
            })
            .collect(),
    )
}

/// The rows of `text` read by `reader` in `lang`.
fn query_rows(graph: &Graph, reader: Reader<'_>, lang: Lang, text: &str) -> Rows {
    let result = match (reader, lang) {
        (_, Lang::Api) => unreachable!("the API runs no queries"),
        (Reader::Session(session), lang) => execute(session, lang, text, HashMap::new()),
        (Reader::At(session, epoch), Lang::Gql) => session.execute_at_epoch(text, epoch),
        #[cfg(feature = "cypher")]
        (Reader::At(session, epoch), Lang::Cypher) => {
            session.set_viewing_epoch(epoch);
            let result = session.execute_cypher(text);
            session.clear_viewing_epoch();
            result
        }
        (Reader::Database, Lang::Gql) => graph.db.execute(text),
        #[cfg(feature = "cypher")]
        (Reader::Database, Lang::Cypher) => graph.db.execute_cypher(text),
    };
    result
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .rows()
        .to_vec()
}

/// What `reader` reads through `read` in `lang`, or `None` when the path is
/// not one of the language's.
fn read(graph: &Graph, reader: Reader<'_>, lang: Lang, read: &Read) -> Option<Rows> {
    match (read, lang) {
        (Read::Api(read), Lang::Api) => read(graph, reader).map(sorted),
        (Read::Api(_), _) | (Read::Query(..), Lang::Api) => None,
        (Read::Query(text, pick), lang) => {
            Some(reduce(&query_rows(graph, reader, lang, text), *pick))
        }
        #[cfg(feature = "cypher")]
        (Read::Cypher(text, pick), Lang::Cypher) => {
            Some(reduce(&query_rows(graph, reader, lang, text), *pick))
        }
        #[cfg(feature = "cypher")]
        (Read::Cypher(..), _) => None,
    }
}

/// Which side of the change a reader must see.
#[derive(Clone, Copy)]
enum Side {
    Before,
    After,
}

/// Reads every path of `paths` that `lang` has as `reader`, and reports
/// whether each reads its `side`, under `prefix / path`.
fn observe(
    report: &mut Report,
    prefix: &str,
    graph: &Graph,
    reader: Reader<'_>,
    lang: Lang,
    paths: &[Path],
    side: Side,
) {
    for path in paths {
        let Some(rows) = read(graph, reader, lang, &path.read) else {
            continue;
        };
        let expected = match side {
            Side::Before => sorted(path.before.clone()),
            Side::After => sorted(path.after.clone()),
        };
        let outcome = if rows == expected {
            Ok(())
        } else {
            Err(format!("expected {expected:?}, read {rows:?}"))
        };
        report.push(format!("{prefix} / {}", path.name), outcome);
    }
}

/// Reads like [`observe`], and panics on any read that does not see `side`:
/// for reads the test relies on, not the guarantee it checks.
fn require(graph: &Graph, reader: Reader<'_>, lang: Lang, paths: &[Path], side: Side, when: &str) {
    let mut report = Report::default();
    observe(&mut report, when, graph, reader, lang, paths, side);
    for (name, outcome) in report.reads {
        if let Err(failure) = outcome {
            panic!("{name}: {failure}");
        }
    }
}

/// The name prefix of the reads of a variant.
fn prefix(scenario: &str, change: Change, lang: Lang, reader: &str, index: Index) -> String {
    format!(
        "{scenario} / {} / {} / {reader} / {}",
        change.name(),
        lang.name(),
        index.name()
    )
}

// ── Direct reads ────────────────────────────────────────────────────

fn node(graph: &Graph, reader: Reader<'_>, node: NodeId) -> Option<grafeo_core::Node> {
    match reader {
        Reader::Session(session) => session.get_node(node),
        Reader::At(_, epoch) => graph.db.get_node_at_epoch(node, epoch).unwrap(),
        Reader::Database => graph.db.get_node(node),
    }
}

fn edge(graph: &Graph, reader: Reader<'_>, edge: EdgeId) -> Option<grafeo_core::Edge> {
    match reader {
        Reader::Session(session) => session.get_edge(edge),
        Reader::At(_, epoch) => graph.db.get_edge_at_epoch(edge, epoch).unwrap(),
        Reader::Database => graph.db.get_edge(edge),
    }
}

/// A node's value of `key`: no row without the node, `NULL` without the
/// value.
fn node_value(of: NodeId, key: &'static str) -> Read {
    api(move |graph, reader| {
        Some(
            node(graph, reader, of)
                .map(|found| vec![found.get_property(key).cloned().unwrap_or(Value::Null)])
                .into_iter()
                .collect(),
        )
    })
}

fn node_labels(of: NodeId) -> Read {
    api(move |graph, reader| {
        Some(
            node(graph, reader, of)
                .map(|found| {
                    let labels: Vec<Value> = found
                        .labels
                        .iter()
                        .map(|label| Value::from(label.as_str()))
                        .collect();
                    vec![sorted_list(Value::List(labels.into()))]
                })
                .into_iter()
                .collect(),
        )
    })
}

fn edge_value(of: EdgeId, key: &'static str) -> Read {
    api(move |graph, reader| {
        Some(
            edge(graph, reader, of)
                .map(|found| vec![found.get_property(key).cloned().unwrap_or(Value::Null)])
                .into_iter()
                .collect(),
        )
    })
}

/// The `(neighbor, edge)` pairs of a node in `direction`; no such read at an
/// epoch.
fn node_edges(of: NodeId, direction: Direction) -> Read {
    api(move |graph, reader| {
        let edges = match reader {
            Reader::Session(session) => session.get_node_edges(of, direction, &[]),
            Reader::At(..) => return None,
            Reader::Database => graph.db.get_node_edges(of, direction, &[]),
        };
        Some(
            edges
                .into_iter()
                .map(|(neighbor, edge)| vec![id(neighbor), edge_id(edge)])
                .collect(),
        )
    })
}

/// The nodes with `age`; no such read at an epoch.
fn nodes_of_age(age: i64) -> Read {
    api(move |graph, reader| {
        let found = match reader {
            Reader::Session(session) => session.find_nodes_by_property("age", &Value::Int64(age)),
            Reader::At(..) => return None,
            Reader::Database => graph.db.find_nodes_by_property("age", &Value::Int64(age)),
        };
        Some(nodes(&found))
    })
}

/// A read of the database only (counts, labels, searches): no such read
/// by a session.
fn database_read(read: impl Fn(&GrafeoDB) -> Rows + Send + Sync + 'static) -> Read {
    api(move |graph, reader| matches!(reader, Reader::Database).then(|| read(&graph.db)))
}

#[cfg(feature = "text-index")]
fn text_search(words: &'static str) -> Read {
    database_read(move |db| {
        let found: Vec<NodeId> = db
            .text_search("Person", "bio", words, 5, None)
            .unwrap()
            .into_iter()
            .map(|(node, _)| node)
            .collect();
        nodes(&found)
    })
}

#[cfg(feature = "vector-index")]
fn vector_search(near: [f32; 3]) -> Read {
    database_read(move |db| {
        let found: Vec<NodeId> = db
            .vector_search("Person", "emb", &near, 1, None, None)
            .unwrap()
            .into_iter()
            .map(|(node, _)| node)
            .collect();
        nodes(&found)
    })
}

// ── The paths of each change ────────────────────────────────────────

/// The paths that read Alix's age, `old` before the change and `new` after.
fn age_paths(graph: &Graph, old: i64, new: i64) -> Vec<Path> {
    let alix = graph.alix;
    let paths = vec![
        path(
            "n.age",
            query("MATCH (n:Person {name: 'Alix'}) RETURN n.age", Pick::Rows),
            ints(&[old]),
            ints(&[new]),
        ),
        path(
            "WHERE n.age = old",
            query(
                &format!("MATCH (n:Person) WHERE n.age = {old} RETURN n.name"),
                Pick::Rows,
            ),
            names(&["Alix"]),
            none(),
        ),
        path(
            "WHERE n.age = new",
            query(
                &format!("MATCH (n:Person) WHERE n.age = {new} RETURN n.name"),
                Pick::Rows,
            ),
            none(),
            names(&["Alix"]),
        ),
        path(
            "{age: old}",
            query(
                &format!("MATCH (n:Person {{age: {old}}}) RETURN n.name"),
                Pick::Rows,
            ),
            names(&["Alix"]),
            none(),
        ),
        path(
            "{age: new}",
            query(
                &format!("MATCH (n:Person {{age: {new}}}) RETURN n.name"),
                Pick::Rows,
            ),
            none(),
            names(&["Alix"]),
        ),
        path(
            "IN list",
            query(
                &format!("MATCH (n:Person) WHERE n.age IN [{old}, 1] RETURN n.name"),
                Pick::Rows,
            ),
            names(&["Alix"]),
            none(),
        ),
        path(
            "range",
            query(
                &format!(
                    "MATCH (n:Person) WHERE n.age >= {} AND n.age <= {old} RETURN n.name",
                    old - 1
                ),
                Pick::Rows,
            ),
            names(&["Alix"]),
            none(),
        ),
        path(
            "sum(n.age)",
            query("MATCH (n:Person) RETURN sum(n.age)", Pick::Rows),
            ints(&[old + 40]),
            ints(&[new + 40]),
        ),
        path(
            "RETURN n",
            query("MATCH (n:Person {name: 'Alix'}) RETURN n", Pick::Key("age")),
            ints(&[old]),
            ints(&[new]),
        ),
        path(
            "properties(n)",
            query(
                "MATCH (n:Person {name: 'Alix'}) RETURN properties(n)",
                Pick::Key("age"),
            ),
            ints(&[old]),
            ints(&[new]),
        ),
        path(
            "get_node",
            node_value(alix, "age"),
            ints(&[old]),
            ints(&[new]),
        ),
        path(
            "find_nodes_by_property old",
            nodes_of_age(old),
            nodes(&[alix]),
            none(),
        ),
        path(
            "find_nodes_by_property new",
            nodes_of_age(new),
            none(),
            nodes(&[alix]),
        ),
    ];
    #[cfg(feature = "cypher")]
    let paths = {
        let mut paths = paths;
        paths.push(path(
            "n {.age}",
            Read::Cypher(
                "MATCH (n:Person {name: 'Alix'}) RETURN n {.age}".to_string(),
                Pick::Key("age"),
            ),
            ints(&[old]),
            ints(&[new]),
        ));
        paths
    };
    paths
}

fn edge_value_paths(graph: &Graph) -> Vec<Path> {
    vec![
        path(
            "r.since",
            query(
                "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() RETURN r.since",
                Pick::Rows,
            ),
            ints(&[2020]),
            ints(&[2099]),
        ),
        path(
            "WHERE r.since = old",
            query(
                "MATCH (:Person)-[r:KNOWS]->(b) WHERE r.since = 2020 RETURN b.name",
                Pick::Rows,
            ),
            names(&["Gus"]),
            none(),
        ),
        path(
            "WHERE r.since = new",
            query(
                "MATCH (:Person)-[r:KNOWS]->(b) WHERE r.since = 2099 RETURN b.name",
                Pick::Rows,
            ),
            none(),
            names(&["Gus"]),
        ),
        path(
            "{since: old}",
            query(
                "MATCH (:Person)-[r:KNOWS {since: 2020}]->(b) RETURN b.name",
                Pick::Rows,
            ),
            names(&["Gus"]),
            none(),
        ),
        path(
            "RETURN r",
            query("MATCH (:Person)-[r:KNOWS]->() RETURN r", Pick::Key("since")),
            ints(&[2020]),
            ints(&[2099]),
        ),
        path(
            "properties(r)",
            query(
                "MATCH (:Person)-[r:KNOWS]->() RETURN properties(r)",
                Pick::Key("since"),
            ),
            ints(&[2020]),
            ints(&[2099]),
        ),
        path(
            "get_edge",
            edge_value(graph.knows, "since"),
            ints(&[2020]),
            ints(&[2099]),
        ),
    ]
}

fn label_paths(graph: &Graph) -> Vec<Path> {
    let alix = graph.alix;
    vec![
        path(
            "label scan of the new label",
            query("MATCH (n:Admin) RETURN n.name", Pick::Rows),
            none(),
            names(&["Alix"]),
        ),
        path(
            "label scan of the old label",
            query("MATCH (n:Person) RETURN n.name", Pick::Rows),
            names(&["Alix", "Gus"]),
            names(&["Gus"]),
        ),
        path(
            "count of the old label",
            query("MATCH (n:Person) RETURN count(n)", Pick::Rows),
            ints(&[2]),
            ints(&[1]),
        ),
        path(
            "label check of the old label",
            query("MATCH (n) WHERE n:Person RETURN n.name", Pick::Rows),
            names(&["Alix", "Gus"]),
            names(&["Gus"]),
        ),
        path(
            "label check of the new label",
            query("MATCH (n) WHERE n:Admin RETURN n.name", Pick::Rows),
            none(),
            names(&["Alix"]),
        ),
        path(
            "label union",
            query(
                "MATCH (n) WHERE n:Admin OR n:Manager RETURN n.name",
                Pick::Rows,
            ),
            none(),
            names(&["Alix"]),
        ),
        path(
            "labels(n)",
            query("MATCH (n {name: 'Alix'}) RETURN labels(n)", Pick::Rows),
            label_list(&["Person"]),
            label_list(&["Admin"]),
        ),
        path(
            "label and key",
            query("MATCH (n:Person {age: 30}) RETURN n.name", Pick::Rows),
            names(&["Alix"]),
            none(),
        ),
        path(
            "RETURN n",
            query("MATCH (n {name: 'Alix'}) RETURN n", Pick::Labels),
            label_list(&["Person"]),
            label_list(&["Admin"]),
        ),
        path(
            "get_node",
            node_labels(alix),
            label_list(&["Person"]),
            label_list(&["Admin"]),
        ),
        path(
            "get_node_labels",
            database_read(move |db| {
                let labels: Vec<Value> = db
                    .get_node_labels(alix)
                    .unwrap_or_default()
                    .into_iter()
                    .map(Value::from)
                    .collect();
                vec![vec![sorted_list(Value::List(labels.into()))]]
            }),
            label_list(&["Person"]),
            label_list(&["Admin"]),
        ),
    ]
}

fn node_delete_paths(graph: &Graph, index: Index) -> Vec<Path> {
    let (alix, gus, knows) = (graph.alix, graph.gus, graph.knows);
    let paths = vec![
        path(
            "label scan",
            query("MATCH (n:Person) RETURN n.name", Pick::Rows),
            names(&["Alix", "Gus"]),
            names(&["Alix"]),
        ),
        path(
            "scan",
            query("MATCH (n) RETURN n.name", Pick::Rows),
            names(&["Alix", "Gus"]),
            names(&["Alix"]),
        ),
        path(
            "count(n)",
            query("MATCH (n) RETURN count(n)", Pick::Rows),
            ints(&[2]),
            ints(&[1]),
        ),
        path(
            "n.age",
            query("MATCH (n:Person {name: 'Gus'}) RETURN n.age", Pick::Rows),
            ints(&[40]),
            none(),
        ),
        path(
            "WHERE n.age = 40",
            query(
                "MATCH (n:Person) WHERE n.age = 40 RETURN n.name",
                Pick::Rows,
            ),
            names(&["Gus"]),
            none(),
        ),
        path(
            "{age: 40}",
            query("MATCH (n:Person {age: 40}) RETURN n.name", Pick::Rows),
            names(&["Gus"]),
            none(),
        ),
        path(
            "labels(n)",
            query("MATCH (n {name: 'Gus'}) RETURN labels(n)", Pick::Rows),
            label_list(&["Person"]),
            none(),
        ),
        path(
            "expand",
            query(
                "MATCH (:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN b.name",
                Pick::Rows,
            ),
            names(&["Gus"]),
            none(),
        ),
        path(
            "count(r)",
            query("MATCH (:Person)-[r:KNOWS]->() RETURN count(r)", Pick::Rows),
            ints(&[1]),
            ints(&[0]),
        ),
        path("get_node", node_value(gus, "age"), ints(&[40]), none()),
        path(
            "get_edge",
            edge_value(knows, "since"),
            ints(&[2020]),
            none(),
        ),
        path(
            "get_node_edges",
            node_edges(alix, Direction::Outgoing),
            vec![vec![id(gus), edge_id(knows)]],
            none(),
        ),
        path(
            "find_nodes_by_property",
            nodes_of_age(40),
            nodes(&[gus]),
            none(),
        ),
        path(
            "node_count",
            database_read(|db| ints(&[i64::try_from(db.node_count()).unwrap()])),
            ints(&[2]),
            ints(&[1]),
        ),
        path(
            "edge_count",
            database_read(|db| ints(&[i64::try_from(db.edge_count()).unwrap()])),
            ints(&[1]),
            ints(&[0]),
        ),
    ];
    let searches = match index {
        Index::None | Index::Property => Vec::new(),
        #[cfg(feature = "text-index")]
        Index::Text => vec![
            path(
                "text_match",
                query(
                    "MATCH (n:Person) WHERE text_match(n.bio, 'Berlin') RETURN n.name",
                    Pick::Rows,
                ),
                names(&["Gus"]),
                none(),
            ),
            path(
                "grafeo.search.text",
                query(
                    "CALL grafeo.search.text('Person', 'bio', 'Berlin', 5)",
                    Pick::First,
                ),
                nodes(&[gus]),
                none(),
            ),
            path("text_search", text_search("Berlin"), nodes(&[gus]), none()),
        ],
        #[cfg(feature = "vector-index")]
        Index::Vector => vec![
            path(
                "top-k by similarity",
                query(
                    "MATCH (n:Person) RETURN n.name \
                     ORDER BY cosine_similarity(n.emb, [0.6, 0.8, 0.0]) DESC LIMIT 1",
                    Pick::Rows,
                ),
                names(&["Gus"]),
                names(&["Alix"]),
            ),
            path(
                "grafeo.search.vector",
                query(
                    "CALL grafeo.search.vector('Person', 'emb', [0.6, 0.8, 0.0], 1)",
                    Pick::First,
                ),
                nodes(&[gus]),
                nodes(&[alix]),
            ),
            path(
                "vector_search",
                vector_search([0.6, 0.8, 0.0]),
                nodes(&[gus]),
                nodes(&[alix]),
            ),
        ],
    };
    paths.into_iter().chain(searches).collect()
}

fn edge_delete_paths(graph: &Graph) -> Vec<Path> {
    let (alix, gus, knows) = (graph.alix, graph.gus, graph.knows);
    vec![
        path(
            "count(r)",
            query("MATCH (:Person)-[r:KNOWS]->() RETURN count(r)", Pick::Rows),
            ints(&[1]),
            ints(&[0]),
        ),
        path(
            "expand",
            query(
                "MATCH (:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN b.name",
                Pick::Rows,
            ),
            names(&["Gus"]),
            none(),
        ),
        path(
            "incoming expand",
            query(
                "MATCH (:Person {name: 'Gus'})<-[:KNOWS]-(a) RETURN a.name",
                Pick::Rows,
            ),
            names(&["Alix"]),
            none(),
        ),
        path(
            "r.since",
            query("MATCH (:Person)-[r:KNOWS]->() RETURN r.since", Pick::Rows),
            ints(&[2020]),
            none(),
        ),
        path(
            "get_edge",
            edge_value(knows, "since"),
            ints(&[2020]),
            none(),
        ),
        path(
            "get_node_edges",
            node_edges(alix, Direction::Outgoing),
            vec![vec![id(gus), edge_id(knows)]],
            none(),
        ),
        path(
            "get_node_edges incoming",
            node_edges(gus, Direction::Incoming),
            vec![vec![id(alix), edge_id(knows)]],
            none(),
        ),
        path(
            "edge_count",
            database_read(|db| ints(&[i64::try_from(db.edge_count()).unwrap()])),
            ints(&[1]),
            ints(&[0]),
        ),
    ]
}

#[cfg(feature = "text-index")]
fn text_paths(graph: &Graph) -> Vec<Path> {
    let alix = graph.alix;
    vec![
        path(
            "text_match of the old value",
            query(
                "MATCH (n:Person) WHERE text_match(n.bio, 'Prague') RETURN n.name",
                Pick::Rows,
            ),
            names(&["Alix"]),
            none(),
        ),
        path(
            "text_match of the new value",
            query(
                "MATCH (n:Person) WHERE text_match(n.bio, 'Paris') RETURN n.name",
                Pick::Rows,
            ),
            none(),
            names(&["Alix"]),
        ),
        path(
            "grafeo.search.text of the old value",
            query(
                "CALL grafeo.search.text('Person', 'bio', 'Prague', 5)",
                Pick::First,
            ),
            nodes(&[alix]),
            none(),
        ),
        path(
            "grafeo.search.text of the new value",
            query(
                "CALL grafeo.search.text('Person', 'bio', 'Paris', 5)",
                Pick::First,
            ),
            none(),
            nodes(&[alix]),
        ),
        path(
            "text_search of the old value",
            text_search("Prague"),
            nodes(&[alix]),
            none(),
        ),
        path(
            "text_search of the new value",
            text_search("Paris"),
            none(),
            nodes(&[alix]),
        ),
        path(
            "n.bio",
            query("MATCH (n:Person {name: 'Alix'}) RETURN n.bio", Pick::Rows),
            names(&["lives in Prague"]),
            names(&["lives in Paris"]),
        ),
    ]
}

#[cfg(feature = "vector-index")]
fn vector_paths(graph: &Graph, index: Index) -> Vec<Path> {
    let (alix, gus) = (graph.alix, graph.gus);
    let mut paths = vec![path(
        "top-k by similarity",
        query(
            "MATCH (n:Person) RETURN n.name \
             ORDER BY cosine_similarity(n.emb, [1.0, 0.0, 0.0]) DESC LIMIT 1",
            Pick::Rows,
        ),
        names(&["Alix"]),
        names(&["Gus"]),
    )];
    if index == Index::Vector {
        paths.extend([
            path(
                "grafeo.search.vector",
                query(
                    "CALL grafeo.search.vector('Person', 'emb', [1.0, 0.0, 0.0], 1)",
                    Pick::First,
                ),
                nodes(&[alix]),
                nodes(&[gus]),
            ),
            path(
                "vector_search",
                vector_search([1.0, 0.0, 0.0]),
                nodes(&[alix]),
                nodes(&[gus]),
            ),
        ]);
    }
    paths
}

fn paths(graph: &Graph, change: Change, index: Index) -> Vec<Path> {
    match change {
        Change::NodeValue => age_paths(graph, 30, 99),
        Change::EdgeValue => edge_value_paths(graph),
        Change::Label => label_paths(graph),
        Change::NodeDelete => node_delete_paths(graph, index),
        Change::EdgeDelete => edge_delete_paths(graph),
        #[cfg(feature = "text-index")]
        Change::Text => text_paths(graph),
        #[cfg(feature = "vector-index")]
        Change::Vector => vector_paths(graph, index),
    }
}

/// Every change, with each of its indexes, in each language.
fn variants() -> Vec<(Change, Index, Lang)> {
    let mut variants = Vec::new();
    for change in Change::all() {
        for index in change.indexes() {
            for lang in languages() {
                variants.push((change, index, lang));
            }
        }
    }
    variants
}

// ── G1, G2, G3, G6: an open transaction's writes ────────────────────

/// G1 (values), G2 (labels), G3 (deletes) and G6 (indexes): while a
/// transaction's write is open, no other reader sees it: not another
/// session, not a transaction begun after the write, not a read at the
/// published epoch, not the database's own reads, through no path and no
/// index.
#[test]
fn g1_g2_g3_g6_an_open_write_is_invisible_to_every_other_reader() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let scenario = match change {
            Change::NodeValue | Change::EdgeValue if index == Index::None => "G1 open write",
            Change::Label => "G2 open write",
            Change::NodeDelete | Change::EdgeDelete => "G3 open write",
            _ => "G6 open write",
        };
        let graph = people(index);
        let paths = paths(&graph, change, index);
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        let epoch = graph.db.current_epoch();
        let other = graph.db.session();
        let mut snapshot = graph.db.session();
        snapshot.begin_transaction().unwrap();
        for (reader, name) in [
            (Reader::Session(&other), "other session"),
            (
                Reader::Session(&snapshot),
                "transaction begun after the write",
            ),
            (Reader::At(&other, epoch), "at the published epoch"),
            (Reader::Database, "database"),
        ] {
            let prefix = prefix(scenario, change, lang, name, index);
            observe(
                &mut report,
                &prefix,
                &graph,
                reader,
                lang,
                &paths,
                Side::Before,
            );
        }
        snapshot.rollback().unwrap();
        writer.rollback().unwrap();
    }
    report.finish();
}

// ── G4, G5: repeatable reads ────────────────────────────────────────

/// G4 (repeatable reads) and G5 (index reads match the snapshot): a
/// transaction reads the committed state, another session commits the
/// change, and the transaction reads the state of its snapshot again
/// through every path and index; so does a read at the epoch before the
/// commit.
#[test]
fn g4_g5_a_snapshot_reads_the_same_after_another_commit() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let scenario = if index == Index::None && !change.is_search() {
            "G4 commit after the snapshot"
        } else {
            "G5 commit after the snapshot"
        };
        let graph = people(index);
        let paths = paths(&graph, change, index);
        let mut snapshot = graph.db.session();
        snapshot.begin_transaction().unwrap();
        let before = graph.db.current_epoch();
        let past = graph.db.session();
        require(
            &graph,
            Reader::Session(&snapshot),
            lang,
            &paths,
            Side::Before,
            "the snapshot's first read",
        );
        change.write(&graph.db.session(), &graph, lang);
        assert!(graph.db.current_epoch() > before, "the change committed");
        for (reader, name) in [
            (
                Reader::Session(&snapshot),
                "transaction begun before the commit",
            ),
            (Reader::At(&past, before), "at the epoch before the commit"),
        ] {
            let prefix = prefix(scenario, change, lang, name, index);
            observe(
                &mut report,
                &prefix,
                &graph,
                reader,
                lang,
                &paths,
                Side::Before,
            );
        }
        snapshot.rollback().unwrap();
    }
    report.finish();
}

// ── G7: rollback restores ───────────────────────────────────────────

/// G7: after a rollback every reader, the writer's session included, reads
/// the state before the transaction through every path and index.
#[test]
fn g7_a_rollback_restores_every_path() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let graph = people(index);
        let paths = paths(&graph, change, index);
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        writer.rollback().unwrap();
        let epoch = graph.db.current_epoch();
        let other = graph.db.session();
        for (reader, name) in [
            (Reader::Session(&writer), "the writer"),
            (Reader::Session(&other), "other session"),
            (Reader::At(&other, epoch), "at the published epoch"),
            (Reader::Database, "database"),
        ] {
            let prefix = prefix("G7 rollback", change, lang, name, index);
            observe(
                &mut report,
                &prefix,
                &graph,
                reader,
                lang,
                &paths,
                Side::Before,
            );
        }
    }
    report.finish();
}

/// G7: a rollback to a savepoint taken before the change restores the
/// state at the savepoint for the transaction itself, and its commit then
/// publishes nothing of the change.
#[test]
fn g7_a_savepoint_rollback_restores_every_path() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let graph = people(index);
        let paths = paths(&graph, change, index);
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        writer.savepoint("before").unwrap();
        change.write(&writer, &graph, lang);
        writer.rollback_to_savepoint("before").unwrap();
        let prefix_of = |name| prefix("G7 savepoint rollback", change, lang, name, index);
        observe(
            &mut report,
            &prefix_of("the transaction"),
            &graph,
            Reader::Session(&writer),
            lang,
            &paths,
            Side::Before,
        );
        writer.commit().unwrap();
        let other = graph.db.session();
        observe(
            &mut report,
            &prefix_of("other session after the commit"),
            &graph,
            Reader::Session(&other),
            lang,
            &paths,
            Side::Before,
        );
    }
    report.finish();
}

/// G7 and G8: Alix's age set to 50, a savepoint, then 99, and a rollback to
/// the savepoint: the transaction reads 50 through every path and index, and
/// after its commit so does every reader.
#[test]
fn g7_g8_a_savepoint_rollback_keeps_the_value_written_before_it() {
    let mut report = Report::default();
    for index in [Index::None, Index::Property] {
        for lang in languages() {
            let graph = people(index);
            let paths = age_paths(&graph, 30, 50);
            let mut writer = graph.db.session();
            writer.begin_transaction().unwrap();
            let set_age = |age: i64| match lang {
                Lang::Api => writer
                    .set_node_property(graph.alix, "age", Value::Int64(age))
                    .unwrap(),
                lang => {
                    let statement = format!("MATCH (n:Person {{name: 'Alix'}}) SET n.age = {age}");
                    execute(&writer, lang, &statement, HashMap::new()).unwrap();
                }
            };
            set_age(50);
            writer.savepoint("at 50").unwrap();
            set_age(99);
            writer.rollback_to_savepoint("at 50").unwrap();
            let prefix_of = |name| {
                prefix(
                    "G7 savepoint rollback",
                    Change::NodeValue,
                    lang,
                    name,
                    index,
                )
            };
            observe(
                &mut report,
                &format!("{} at 50", prefix_of("the transaction")),
                &graph,
                Reader::Session(&writer),
                lang,
                &paths,
                Side::After,
            );
            writer.commit().unwrap();
            let other = graph.db.session();
            for (reader, name) in [
                (Reader::Session(&other), "other session after the commit"),
                (Reader::Database, "database after the commit"),
            ] {
                observe(
                    &mut report,
                    &format!("{} at 50", prefix_of(name)),
                    &graph,
                    reader,
                    lang,
                    &paths,
                    Side::After,
                );
            }
        }
    }
    report.finish();
}

// ── G8: read your own writes ────────────────────────────────────────

/// G8: inside the writing transaction every path and index sees its own
/// change.
#[test]
fn g8_a_transaction_reads_its_own_writes_on_every_path() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let graph = people(index);
        let paths = paths(&graph, change, index);
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        let prefix = prefix("G8 own writes", change, lang, "the transaction", index);
        observe(
            &mut report,
            &prefix,
            &graph,
            Reader::Session(&writer),
            lang,
            &paths,
            Side::After,
        );
        writer.rollback().unwrap();
    }
    report.finish();
}

/// Once the change commits, every new reader sees it through every path and
/// index: the other side of the guarantees above, and the check that each
/// path reads what the change wrote.
#[test]
fn a_commit_publishes_the_change_on_every_path() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let graph = people(index);
        let paths = paths(&graph, change, index);
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        writer.commit().unwrap();
        let epoch = graph.db.current_epoch();
        let other = graph.db.session();
        let mut snapshot = graph.db.session();
        snapshot.begin_transaction().unwrap();
        for (reader, name) in [
            (Reader::Session(&other), "other session"),
            (
                Reader::Session(&snapshot),
                "transaction begun after the commit",
            ),
            (Reader::At(&other, epoch), "at the published epoch"),
            (Reader::Database, "database"),
        ] {
            let prefix = prefix("commit publishes", change, lang, name, index);
            observe(
                &mut report,
                &prefix,
                &graph,
                reader,
                lang,
                &paths,
                Side::After,
            );
        }
        snapshot.rollback().unwrap();
    }
    report.finish();
}

// ── G9: no lost update from an aborted writer ───────────────────────

/// One `:Counter {n: 0}` node.
fn counter(index: Index) -> (Arc<GrafeoDB>, NodeId) {
    let db = GrafeoDB::new_in_memory();
    let node = db
        .create_node_with_props(&["Counter"], [("n", Value::Int64(0))])
        .unwrap();
    if index == Index::Property {
        db.create_property_index("n").unwrap();
    }
    (Arc::new(db), node)
}

fn read_counter(session: &Session, lang: Lang, node: NodeId) -> i64 {
    let value = match lang {
        Lang::Api => session.get_node_property(node, "n"),
        lang => execute(
            session,
            lang,
            "MATCH (c:Counter) RETURN c.n",
            HashMap::new(),
        )
        .unwrap()
        .rows()
        .first()
        .map(|row| row[0].clone()),
    };
    match value {
        Some(Value::Int64(n)) => n,
        other => panic!("the counter reads {other:?}"),
    }
}

fn write_counter(session: &Session, lang: Lang, node: NodeId, n: i64) -> Result<()> {
    match lang {
        Lang::Api => session.set_node_property(node, "n", Value::Int64(n)),
        lang => execute(
            session,
            lang,
            &format!("MATCH (c:Counter) SET c.n = {n}"),
            HashMap::new(),
        )
        .map(drop),
    }
}

/// G9, step by step: a transaction increments the counter and stays open,
/// a second reads the counter, the first rolls back, the second writes what
/// it read plus one and commits. The counter must be 1: the second must not
/// have read the increment that never committed.
#[test]
fn g9_an_aborted_increment_is_not_counted() {
    let mut report = Report::default();
    for index in [Index::None, Index::Property] {
        for lang in languages() {
            let (db, node) = counter(index);
            let mut first = db.session();
            first.begin_transaction().unwrap();
            write_counter(&first, lang, node, 1).unwrap();
            let mut second = db.session();
            second.begin_transaction().unwrap();
            let read = read_counter(&second, lang, node);
            first.rollback().unwrap();
            write_counter(&second, lang, node, read + 1).unwrap();
            second.commit().unwrap();
            let after = read_counter(&db.session(), lang, node);
            let outcome = if after == 1 {
                Ok(())
            } else {
                Err(format!(
                    "the counter is {after}: the second read {read} from the open increment"
                ))
            };
            report.push(
                format!(
                    "G9 aborted increment / counter / {} / two transactions / {} / the counter",
                    lang.name(),
                    index.name()
                ),
                outcome,
            );
        }
    }
    report.finish();
}

/// Whether `error` is a write conflict, after which the increment retries.
fn is_conflict(error: &Error) -> bool {
    matches!(
        error,
        Error::Transaction(
            TransactionError::WriteConflict(_)
                | TransactionError::Conflict
                | TransactionError::SerializationFailure(_)
        )
    )
}

/// Increments the counter in a transaction that reads it and writes it
/// plus one, retried until it commits; returns the conflicts it retried.
fn increment(db: &GrafeoDB, lang: Lang, node: NodeId) -> usize {
    let mut conflicts = 0;
    loop {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let n = read_counter(&session, lang, node);
        let outcome = write_counter(&session, lang, node, n + 1).and_then(|()| session.commit());
        match outcome {
            Ok(()) => return conflicts,
            Err(error) if is_conflict(&error) => {
                conflicts += 1;
                if session.in_transaction() {
                    session.rollback().unwrap();
                }
            }
            Err(error) => panic!("an increment failed: {error}"),
        }
    }
}

/// G9, threaded (issue comment of 2026-09-29): four threads each increment
/// one counter 100 times, each increment a transaction that reads the
/// counter, writes it plus one and retries on a conflict. The counter ends
/// at exactly 400, in each of thirty rounds: about half the rounds of
/// today's store end at 400 by chance (the others above or below it),
/// thirty in a row do not.
#[test]
fn g9_concurrent_increments_lose_and_add_nothing() {
    const THREADS: usize = 4;
    const INCREMENTS: usize = 100;
    const ROUNDS: usize = 30;
    let mut report = Report::default();
    for lang in query_languages() {
        let mut finals = Vec::new();
        let mut conflicts = 0;
        for _round in 0..ROUNDS {
            let (db, node) = counter(Index::None);
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let db = Arc::clone(&db);
                    std::thread::spawn(move || {
                        (0..INCREMENTS)
                            .map(|_| increment(&db, lang, node))
                            .sum::<usize>()
                    })
                })
                .collect();
            for handle in handles {
                conflicts += handle.join().expect("an incrementing thread panicked");
            }
            finals.push(read_counter(&db.session(), lang, node));
        }
        let expected = i64::try_from(THREADS * INCREMENTS).unwrap();
        let outcome = if finals.iter().all(|n| *n == expected) {
            Ok(())
        } else {
            Err(format!(
                "the counter ends at {finals:?} after {conflicts} retried conflicts, expected {expected}"
            ))
        };
        report.push(
            format!(
                "G9 concurrent increments / counter / {} / four threads / no index / the counter",
                lang.name()
            ),
            outcome,
        );
    }
    report.finish();
}

/// G9, threaded with a forced schedule (the Python binding's counter test):
/// two threads increment one counter in a fixed order of their steps, so
/// the outcome does not depend on how the threads are scheduled (the free
/// race above can interleave rarely and pass by chance on a store without
/// isolation). In each round:
///
/// 1. the writer begins, reads the counter and writes it plus one;
/// 2. the reader begins and reads the counter;
/// 3. the writer gives its attempt up (rolls back, as a conflict does);
/// 4. the reader writes what it read plus one, and commits;
/// 5. the writer increments again.
///
/// Each round commits two increments: the reader reads the committed value,
/// never the writer's, so the counter ends at twice the rounds.
#[test]
fn g9_increments_in_a_forced_schedule_count_each_commit_once() {
    use std::sync::mpsc::{Receiver, Sender, channel};

    const ROUNDS: usize = 25;
    let mut report = Report::default();
    for lang in query_languages() {
        let (db, node) = counter(Index::None);
        // A step the other thread waits for: one message per round.
        let (written, wait_written) = channel::<()>();
        let (read, wait_read) = channel::<()>();
        let (given_up, wait_given_up) = channel::<()>();
        let (committed, wait_committed) = channel::<()>();
        let (incremented, wait_incremented) = channel::<()>();
        let step = |to: &Sender<()>| to.send(()).expect("the other thread is gone");
        let wait = |from: &Receiver<()>| from.recv().expect("the other thread is gone");

        std::thread::scope(|scope| {
            let writer = {
                let db = Arc::clone(&db);
                scope.spawn(move || {
                    for _ in 0..ROUNDS {
                        let mut session = db.session();
                        session.begin_transaction().unwrap();
                        let n = read_counter(&session, lang, node);
                        write_counter(&session, lang, node, n + 1).unwrap();
                        step(&written);
                        wait(&wait_read);
                        session.rollback().unwrap();
                        step(&given_up);
                        wait(&wait_committed);
                        increment(&db, lang, node);
                        step(&incremented);
                    }
                })
            };
            let reader = {
                let db = Arc::clone(&db);
                scope.spawn(move || {
                    for _ in 0..ROUNDS {
                        wait(&wait_written);
                        let mut session = db.session();
                        session.begin_transaction().unwrap();
                        let n = read_counter(&session, lang, node);
                        step(&read);
                        wait(&wait_given_up);
                        write_counter(&session, lang, node, n + 1).unwrap();
                        session.commit().unwrap();
                        step(&committed);
                        wait(&wait_incremented);
                    }
                })
            };
            writer.join().expect("the writer panicked");
            reader.join().expect("the reader panicked");
        });

        let found = read_counter(&db.session(), lang, node);
        let expected = i64::try_from(2 * ROUNDS).unwrap();
        report.push(
            format!(
                "G9 forced schedule / counter / {} / writer and reader / no index / the counter",
                lang.name()
            ),
            if found == expected {
                Ok(())
            } else {
                Err(format!("the counter ends at {found}, expected {expected}"))
            },
        );
    }
    report.finish();
}

// ── G10: a commit that never publishes ──────────────────────────────

/// The readers of a commit that did not publish, after it failed or while it
/// runs: they all read the state before it.
#[cfg(any(
    feature = "testing-statement-injection",
    all(
        feature = "testing-crash-injection",
        feature = "wal",
        feature = "grafeo-file"
    )
))]
fn observe_unpublished(
    report: &mut Report,
    scenario: &str,
    graph: &Graph,
    change: Change,
    index: Index,
    lang: Lang,
    epoch: EpochId,
) {
    let paths = paths(graph, change, index);
    let other = graph.db.session();
    let mut snapshot = graph.db.session();
    snapshot.begin_transaction().unwrap();
    for (reader, name) in [
        (Reader::Session(&other), "other session"),
        (
            Reader::Session(&snapshot),
            "transaction begun after the commit",
        ),
        (Reader::At(&other, epoch), "at the published epoch"),
        (Reader::Database, "database"),
    ] {
        let prefix = prefix(scenario, change, lang, name, index);
        observe(report, &prefix, graph, reader, lang, &paths, Side::Before);
    }
    snapshot.rollback().unwrap();
}

/// G10: a commit that panics once its versions are stamped never
/// publishes, and no reader sees any of it.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn g10_a_commit_stopped_after_stamping_stays_invisible() {
    use grafeo_common::testing::commit_hook::after_next_commit_stamped;

    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let graph = people(index);
        let before = graph.db.current_epoch();
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        after_next_commit_stamped(|| panic!("injected: the commit stops after stamping"));
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.commit()));
        assert!(unwound.is_err(), "the commit panicked");
        assert_eq!(graph.db.current_epoch(), before, "nothing was published");
        observe_unpublished(
            &mut report,
            "G10 commit stopped after stamping",
            &graph,
            change,
            index,
            lang,
            before,
        );
    }
    report.finish();
}

/// G10: while a commit is stamped and not yet published, readers on other
/// threads read the state before it.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn g10_a_commit_in_progress_is_invisible_until_it_publishes() {
    use grafeo_common::testing::commit_hook::after_next_commit_stamped;
    use std::sync::mpsc;

    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let graph = people(index);
        let before = graph.db.current_epoch();
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        let (sender, during) = mpsc::channel();
        let reading = graph.clone();
        after_next_commit_stamped(move || {
            let seen = std::thread::spawn(move || {
                let mut report = Report::default();
                observe_unpublished(
                    &mut report,
                    "G10 commit in progress",
                    &reading,
                    change,
                    index,
                    lang,
                    before,
                );
                report
            })
            .join()
            .expect("the reads during the commit panicked");
            sender.send(seen).unwrap();
        });
        writer.commit().unwrap();
        report.extend(during.recv().expect("the commit ran the hook"));
    }
    report.finish();
}

/// G10: a commit whose WAL sync fails (`GRAFEO-T010`, the outcome is
/// known at the reopen) stays invisible to every reader until then.
#[cfg(all(
    feature = "testing-crash-injection",
    feature = "wal",
    feature = "grafeo-file"
))]
#[test]
fn g10_a_commit_whose_wal_sync_failed_stays_invisible() {
    use grafeo_common::testing::crash::with_failure_at;
    use grafeo_engine::Config;
    use grafeo_engine::config::{DurabilityMode, StorageFormat};

    /// The failure of the group's sync: the write is the first I/O point.
    const GROUP_SYNC: u64 = 2;

    let mut report = Report::default();
    for (change, index, lang) in variants() {
        let dir = tempfile::tempdir().unwrap();
        let db = GrafeoDB::with_config(
            Config::persistent(dir.path().join("prague.grafeo"))
                .with_storage_format(StorageFormat::Auto)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap();
        let graph = people_in(db, index);
        let before = graph.db.current_epoch();
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        change.write(&writer, &graph, lang);
        let error = with_failure_at(GROUP_SYNC, || writer.commit())
            .expect_err("the commit reports its failed sync");
        assert_eq!(error.error_code().as_str(), "GRAFEO-T010", "{error}");
        assert_eq!(graph.db.current_epoch(), before, "nothing was published");
        observe_unpublished(
            &mut report,
            "G10 WAL sync failed",
            &graph,
            change,
            index,
            lang,
            before,
        );
        drop(writer);
        assert!(graph.db.close().is_err(), "close reports the commit");
    }
    report.finish();
}

// ── G12: deletes take effect at their commit epoch ──────────────────

/// G12: a reader at an epoch between the start of the deleting transaction
/// and its commit still sees the deleted node or edge, through every path;
/// a reader at the commit epoch does not.
#[test]
fn g12_a_delete_takes_effect_at_its_commit_epoch() {
    let mut report = Report::default();
    for (change, index, lang) in variants() {
        if !change.is_delete() {
            continue;
        }
        // History kept: the reads between the start and the commit need it.
        let graph = people_in(
            GrafeoDB::with_config(
                grafeo_engine::Config::in_memory()
                    .with_history_retention(grafeo_engine::HistoryRetention::Forever),
            )
            .unwrap(),
            index,
        );
        let paths = paths(&graph, change, index);
        let mut writer = graph.db.session();
        writer.begin_transaction().unwrap();
        // Another commit moves the epoch on while the deleting transaction
        // is open; it writes a value no path reads.
        graph
            .db
            .execute("MATCH (n:Person {name: 'Alix'}) SET n.city = 'Amsterdam'")
            .unwrap();
        let between = graph.db.current_epoch();
        change.write(&writer, &graph, lang);
        writer.commit().unwrap();
        let committed = graph.db.current_epoch();
        let other = graph.db.session();
        let prefix_of = |name| prefix("G12 delete before its commit", change, lang, name, index);
        observe(
            &mut report,
            &prefix_of("at an epoch between the start and the commit"),
            &graph,
            Reader::At(&other, between),
            lang,
            &paths,
            Side::Before,
        );
        observe(
            &mut report,
            &prefix_of("at the commit epoch"),
            &graph,
            Reader::At(&other, committed),
            lang,
            &paths,
            Side::After,
        );
    }
    report.finish();
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[path = "common/replay.rs"]
mod replay;

/// G12 after a replay: the deleting transaction of
/// [`g12_a_delete_takes_effect_at_its_commit_epoch`] in a database file
/// whose process exits without closing it; the reopen replays the WAL and
/// builds the same history, so a reader at the epoch between the
/// transaction's start and its commit still sees Gus and his edge.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn g12_a_replayed_delete_takes_effect_at_its_commit_epoch() {
    use std::path::{Path, PathBuf};

    /// Where the child notes the epoch between the start and the commit,
    /// and the ids of Alix, Gus and their edge.
    fn notes(db: &GrafeoDB) -> PathBuf {
        db.path().expect("a database file").with_extension("notes")
    }

    let (_dir, db) = replay::reopened_after_crash(
        "g12_a_replayed_delete_takes_effect_at_its_commit_epoch",
        // History kept: the reads between the start and the commit need it
        // (the file stores the setting, so the reopen keeps it too).
        |path: &Path| {
            GrafeoDB::with_config(
                grafeo_engine::Config::persistent(path)
                    .with_history_retention(grafeo_engine::HistoryRetention::Forever),
            )
            .unwrap()
        },
        |db| {
            let (alix, gus, knows) = fill(db, Index::None);
            let mut writer = db.session();
            writer.begin_transaction().unwrap();
            db.execute("MATCH (n:Person {name: 'Alix'}) SET n.city = 'Amsterdam'")
                .unwrap();
            let between = db.current_epoch();
            writer.execute(Change::NodeDelete.statements()[0]).unwrap();
            writer.commit().unwrap();
            let noted = [
                between.as_u64(),
                alix.as_u64(),
                gus.as_u64(),
                knows.as_u64(),
            ]
            .map(|number| number.to_string())
            .join(" ");
            std::fs::write(notes(db), noted).unwrap();
        },
    );
    let noted: Vec<u64> = std::fs::read_to_string(notes(&db))
        .unwrap()
        .split(' ')
        .map(|number| number.parse().unwrap())
        .collect();
    let between = EpochId::new(noted[0]);
    let graph = Graph {
        db: Arc::new(db),
        alix: NodeId::new(noted[1]),
        gus: NodeId::new(noted[2]),
        knows: EdgeId::new(noted[3]),
    };
    let paths = paths(&graph, Change::NodeDelete, Index::None);
    let committed = graph.db.current_epoch();
    assert!(committed > between, "the reopen continues at the commit");
    let other = graph.db.session();
    let mut report = Report::default();
    for lang in languages() {
        let prefix_of = |name| {
            prefix(
                "G12 delete replayed",
                Change::NodeDelete,
                lang,
                name,
                Index::None,
            )
        };
        observe(
            &mut report,
            &prefix_of("at an epoch between the start and the commit"),
            &graph,
            Reader::At(&other, between),
            lang,
            &paths,
            Side::Before,
        );
        observe(
            &mut report,
            &prefix_of("at the commit epoch"),
            &graph,
            Reader::At(&other, committed),
            lang,
            &paths,
            Side::After,
        );
    }
    drop(other);
    graph.db.close().unwrap();
    report.finish();
}

// ── An open transaction's creates (G6, G8) ──────────────────────────

/// The node ids in the first column of `query`'s rows, sorted.
fn first_ids(session: &Session, query: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Int64(id)) => Some(*id),
            _ => None,
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// The node's id as a query returns it.
fn raw(node: NodeId) -> i64 {
    i64::try_from(node.as_u64()).unwrap()
}

/// A node another session created and has not committed is in the text and
/// vector indexes, but no search procedure of another session finds it; its
/// own transaction finds it (the procedures read at the caller's view).
#[cfg(all(feature = "text-index", feature = "vector-index"))]
#[test]
fn search_procedures_skip_an_open_transactions_creates() {
    let graph = people(Index::Text);
    Index::Vector.create(&graph.db);
    let mut writer = graph.db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Person {name: 'Jules', bio: 'lives in Prague', emb: vector([1.0, 0.0, 0.0])})")
        .unwrap();
    let jules = first_ids(&writer, "MATCH (n:Person {name: 'Jules'}) RETURN id(n)");
    assert_eq!(jules.len(), 1, "the writer sees Jules");
    let other = graph.db.session();
    let text = "CALL grafeo.search.text('Person', 'bio', 'Prague', 5)";
    let vector = "CALL grafeo.search.vector('Person', 'emb', [1.0, 0.0, 0.0], 5)";
    let mut with_jules = vec![raw(graph.alix), jules[0]];
    with_jules.sort_unstable();
    assert_eq!(first_ids(&other, text), [raw(graph.alix)], "{text}");
    assert_eq!(first_ids(&writer, text), with_jules, "{text} in the writer");
    let mut everyone = vec![raw(graph.alix), raw(graph.gus)];
    everyone.sort_unstable();
    assert_eq!(first_ids(&other, vector), everyone, "{vector}");
    everyone.push(jules[0]);
    everyone.sort_unstable();
    assert_eq!(
        first_ids(&writer, vector),
        everyone,
        "{vector} in the writer"
    );
    writer.rollback().unwrap();
    graph.db.close().unwrap();
}

/// A `MERGE` without a label or an index finds the node its transaction
/// created before (the candidates are the nodes the transaction sees).
#[test]
fn a_merge_finds_its_transactions_own_node_without_a_label_or_index() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT ({city: 'Amsterdam'})").unwrap();
    session.execute("MERGE ({city: 'Amsterdam'})").unwrap();
    let count = session
        .execute("MATCH (n {city: 'Amsterdam'}) RETURN count(n)")
        .unwrap()
        .rows()[0][0]
        .clone();
    assert_eq!(count, Value::Int64(1));
    session.commit().unwrap();
}
