//! A planner rule changes how a query runs, never the rows it answers.
//!
//! Random multigraphs and random patterns, in GQL, Cypher and SQL/PGQ, three ways:
//! every rule on, rules off in every combination that matters, and a brute-force
//! enumerator written here that shares no code with the engine. A difference fails
//! the test with its seed and a shrunken graph that still shows it.
//!
//! The matching rule of each language is stated once, in [`Language::edges_may_repeat`]:
//! Cypher binds an edge once per `MATCH`; GQL (its match mode is implementation-defined,
//! ISO/IEC 39075 4.11.9, ID086) and SQL/PGQ repeat edges. When the enumerator and the
//! engine disagree the engine or the rule stated here is wrong: settle which, and never
//! bend the enumerator towards what the engine answers.
//!
//! The contract of a rule, comparing it on with the same rule off: when both succeed the
//! rows are the same; rule off failing while rule on succeeds is allowed, and which
//! condition is raised may differ (5.3.2.4, US008); rule off succeeding while rule on fails
//! is a failure of this test. The random queries of the first test raise nothing, so there
//! an error on any side is a difference; the last test has conjuncts that do raise.
//!
//! `GRAFEO_RULES_SEEDS=2000 cargo test -p grafeo-engine --all-features --test
//! planner_rules_results` sweeps more seeds; `GRAFEO_RULES_SEED=17` runs one.

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

/// The seeds CI runs.
const FIXED_SEEDS: u64 = 24;
/// The queries asked of each graph.
const QUERIES_PER_GRAPH: usize = 14;
/// The bindings the enumerator walks for one query before it gives the query up.
const ENUMERATION_BUDGET: usize = 150_000;
/// The rows an answer may hold; a larger one tells no more and costs much.
const ANSWER_BUDGET: usize = 8_000;
/// What a read adds to a variable's count of uses, far above the uses in patterns.
const READ: usize = 1_000;
/// The differences a run reports in full, each shrunk.
const REPORTED: usize = 6;

const LABELS: [&str; 3] = ["Person", "City", "Tag"];
const KINDS: [&str; 3] = ["KNOWS", "LIKES", "VISITS"];

// ---------------------------------------------------------------------------
// A small generator: the same seed gives the same graph and queries everywhere
// ---------------------------------------------------------------------------

struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5851_F42D_4C95_7F2D)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound.max(1) as u64).unwrap()
    }

    fn range(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low + 1)
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.below(from.len())]
    }
}

// ---------------------------------------------------------------------------
// The graph, as this test knows it
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Node {
    key: i64,
    labels: Vec<&'static str>,
    value: Option<i64>,
}

#[derive(Clone, Debug)]
struct Edge {
    key: i64,
    source: i64,
    target: i64,
    kind: &'static str,
    weight: Option<i64>,
}

#[derive(Clone, Debug, Default)]
struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

/// A statement that changes the graph, applied to the engine and to the model alike. One
/// that names what is gone does nothing on either, so a shrunken graph can keep the list.
#[derive(Clone, Debug)]
enum Write {
    AddNode(Node),
    AddEdge(Edge),
    RemoveEdge(i64),
    RemoveNode(i64),
    SetValue(i64, i64),
}

impl Write {
    fn text(&self) -> String {
        match self {
            Self::AddNode(node) => {
                let labels: String = node.labels.iter().fold(String::new(), |mut out, l| {
                    write!(out, ":{l}").unwrap();
                    out
                });
                let value = node.value.map_or(String::new(), |v| format!(", v: {v}"));
                format!("INSERT ({labels} {{k: {}{value}}})", node.key)
            }
            Self::AddEdge(edge) => {
                let weight = edge.weight.map_or(String::new(), |w| format!(", w: {w}"));
                format!(
                    "MATCH (a {{k: {}}}), (b {{k: {}}}) INSERT (a)-[:{} {{k: {}{weight}}}]->(b)",
                    edge.source, edge.target, edge.kind, edge.key
                )
            }
            Self::RemoveEdge(key) => format!("MATCH ()-[e {{k: {key}}}]->() DELETE e"),
            Self::RemoveNode(key) => format!("MATCH (n {{k: {key}}}) DETACH DELETE n"),
            Self::SetValue(key, value) => format!("MATCH (n {{k: {key}}}) SET n.v = {value}"),
        }
    }

    fn apply_to(&self, graph: &mut Graph) {
        let has = |graph: &Graph, key| graph.nodes.iter().any(|n| n.key == key);
        match self {
            Self::AddNode(node) => graph.nodes.push(node.clone()),
            Self::AddEdge(edge) => {
                if has(graph, edge.source) && has(graph, edge.target) {
                    graph.edges.push(edge.clone());
                }
            }
            Self::RemoveEdge(key) => graph.edges.retain(|e| e.key != *key),
            Self::RemoveNode(key) => {
                graph.nodes.retain(|n| n.key != *key);
                graph.edges.retain(|e| e.source != *key && e.target != *key);
            }
            Self::SetValue(key, value) => {
                for node in graph.nodes.iter_mut().filter(|n| n.key == *key) {
                    node.value = Some(*value);
                }
            }
        }
    }
}

impl Graph {
    fn random(random: &mut Random, largest: usize, most_edges: usize) -> Self {
        let count = random.range(1, largest);
        let nodes = (0..count)
            .map(|index| {
                let mut labels = Vec::new();
                for label in LABELS {
                    if random.chance(40) {
                        labels.push(label);
                    }
                }
                Node {
                    key: index as i64,
                    labels,
                    value: random.chance(75).then(|| random.below(5) as i64),
                }
            })
            .collect();
        let mut graph = Self {
            nodes,
            edges: Vec::new(),
        };
        // Some graphs have a hub, and some nodes stay alone.
        let hub = random.chance(40).then(|| random.below(count) as i64);
        let connected = random.range(count.div_ceil(2), count) as i64;
        let wanted = if random.chance(10) {
            random.below(4)
        } else {
            random.range(most_edges / 2, most_edges)
        };
        while graph.edges.len() < wanted {
            let key = graph.edges.len() as i64;
            let pick = |random: &mut Random| {
                if let Some(hub) = hub.filter(|_| random.chance(35)) {
                    hub
                } else {
                    random.below(connected as usize) as i64
                }
            };
            let (source, target) = (pick(random), pick(random));
            let edge = Edge {
                key,
                source,
                target,
                kind: random.pick(&KINDS),
                weight: random.chance(70).then(|| random.below(4) as i64),
            };
            // A pair repeated in this direction or the other, often: multi-edges and
            // two-cycles.
            let again = random.chance(25);
            graph.edges.push(edge.clone());
            if again && graph.edges.len() < wanted {
                let key = graph.edges.len() as i64;
                let (source, target) = if random.chance(50) {
                    (edge.source, edge.target)
                } else {
                    (edge.target, edge.source)
                };
                graph.edges.push(Edge {
                    key,
                    source,
                    target,
                    ..edge
                });
            }
        }
        graph
    }

    /// Writes for a graph: new edges, nodes and values, and things removed.
    fn random_writes(&self, random: &mut Random) -> Vec<Write> {
        let keys: Vec<i64> = self.nodes.iter().map(|n| n.key).collect();
        let mut next_node = 1_000;
        let mut next_edge = 1_000;
        let mut new_nodes = Vec::new();
        let mut writes = Vec::new();
        for _ in 0..random.range(1, 8) {
            match random.below(10) {
                0 => {
                    let node = Node {
                        key: next_node,
                        labels: vec![random.pick(&LABELS)],
                        value: Some(random.below(5) as i64),
                    };
                    new_nodes.push(next_node);
                    next_node += 1;
                    writes.push(Write::AddNode(node));
                }
                1..=4 => {
                    let any = |random: &mut Random, new: &[i64]| {
                        if !new.is_empty() && random.chance(30) {
                            random.pick(new)
                        } else {
                            random.pick(&keys)
                        }
                    };
                    writes.push(Write::AddEdge(Edge {
                        key: next_edge,
                        source: any(random, &new_nodes),
                        target: any(random, &new_nodes),
                        kind: random.pick(&KINDS),
                        weight: Some(random.below(4) as i64),
                    }));
                    next_edge += 1;
                }
                5..=7 if !self.edges.is_empty() => {
                    let edge = &self.edges[random.below(self.edges.len())];
                    writes.push(Write::RemoveEdge(edge.key));
                }
                8 => writes.push(Write::SetValue(random.pick(&keys), random.below(5) as i64)),
                _ => writes.push(Write::RemoveNode(random.pick(&keys))),
            }
        }
        writes
    }

    fn after(&self, writes: &[Write]) -> Self {
        let mut graph = self.clone();
        for write in writes {
            write.apply_to(&mut graph);
        }
        graph
    }

    fn statements(&self) -> String {
        let mut text = String::new();
        for node in &self.nodes {
            let _ = writeln!(text, "{}", Write::AddNode(node.clone()).text());
        }
        for edge in &self.edges {
            let _ = writeln!(text, "{}", Write::AddEdge(edge.clone()).text());
        }
        text
    }
}

// ---------------------------------------------------------------------------
// The queries
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Way {
    Out,
    In,
    Either,
}

#[derive(Clone, Debug, Default)]
struct NodeVariable {
    label: Option<&'static str>,
    /// The label goes on the variable's last use, not its first.
    label_last: bool,
    /// `{k: ...}` or `{v: ...}` on the first use.
    key: Option<i64>,
    value: Option<i64>,
    /// May print as `()` when nothing else reads it.
    anonymous: bool,
}

#[derive(Clone, Debug)]
struct Hop {
    edge: Option<usize>,
    way: Way,
    kinds: Vec<&'static str>,
    weight: Option<i64>,
    /// A quantifier: at least, at most.
    repeat: Option<(usize, usize)>,
    to: usize,
}

#[derive(Clone, Debug)]
struct Path {
    from: usize,
    hops: Vec<Hop>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Compare {
    Equal,
    Different,
    Less,
    AtLeast,
}

impl Compare {
    fn text(self) -> &'static str {
        match self {
            Self::Equal => "=",
            Self::Different => "<>",
            Self::Less => "<",
            Self::AtLeast => ">=",
        }
    }

    fn holds(self, left: i64, right: i64) -> bool {
        match self {
            Self::Equal => left == right,
            Self::Different => left != right,
            Self::Less => left < right,
            Self::AtLeast => left >= right,
        }
    }
}

#[derive(Clone, Debug)]
enum Condition {
    SameNode(usize, usize),
    Value(usize, Compare, i64),
    Key(usize, i64),
    NodeId(usize, i64),
    Weight(usize, Compare, i64),
    SameValue(usize, usize),
    Or(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
    Exists(Vec<Path>),
    /// `100 / n.v > 0`: raises on a value of 0.
    Raises(usize),
}

#[derive(Clone, Debug)]
struct Clause {
    optional: bool,
    paths: Vec<Path>,
    conditions: Vec<Condition>,
    /// The conditions are statements of their own, one after the other, not one `WHERE`.
    stacked: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Column {
    Key(usize),
    Value(usize),
    EdgeKey(usize),
}

#[derive(Clone, Debug)]
enum Answer {
    Rows(Vec<Column>),
    Distinct(Vec<Column>),
    Count,
    CountBy(Column),
    /// Keys only, so that a tie is two equal rows.
    First(Vec<Column>, usize),
}

#[derive(Clone, Debug)]
struct Query {
    variables: Vec<NodeVariable>,
    edge_variables: usize,
    clauses: Vec<Clause>,
    answer: Answer,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Language {
    Gql,
    /// GQL with its constants as parameters, through the prepared plan.
    GqlParameters,
    Cypher,
    Sql,
}

impl Language {
    const ALL: [Self; 4] = [Self::Gql, Self::GqlParameters, Self::Cypher, Self::Sql];

    /// Whether two edge patterns of one `MATCH` may bind the same edge.
    fn edges_may_repeat(self) -> bool {
        self != Self::Cypher
    }
}

/// A query's text in one language, and its parameters.
struct Text {
    query: String,
    parameters: HashMap<String, Value>,
}

impl Text {
    fn constant(&mut self, language: Language, value: i64) -> String {
        if language == Language::GqlParameters {
            let name = format!("p{}", self.parameters.len());
            self.parameters.insert(name.clone(), Value::Int64(value));
            format!("${name}")
        } else {
            value.to_string()
        }
    }
}

impl Column {
    fn text(self) -> String {
        match self {
            Self::Key(node) => format!("n{node}.k"),
            Self::Value(node) => format!("n{node}.v"),
            Self::EdgeKey(edge) => format!("e{edge}.k"),
        }
    }
}

impl Query {
    fn columns(&self) -> Vec<Column> {
        match &self.answer {
            Answer::Rows(columns) | Answer::Distinct(columns) | Answer::First(columns, _) => {
                columns.clone()
            }
            Answer::Count => Vec::new(),
            Answer::CountBy(column) => vec![*column],
        }
    }

    /// Whether SQL/PGQ can say it: one `GRAPH_TABLE` of one clause, no subquery.
    fn fits_sql(&self) -> bool {
        fn plain(condition: &Condition) -> bool {
            match condition {
                Condition::Exists(_) => false,
                Condition::Or(left, right) => plain(left) && plain(right),
                Condition::Not(inner) => plain(inner),
                _ => true,
            }
        }
        self.clauses.len() == 1
            && !self.clauses[0].stacked
            && self.clauses[0].conditions.iter().all(plain)
    }

    fn uses_of_nodes(&self, language: Language) -> Vec<usize> {
        let mut uses = vec![0; self.variables.len()];
        fn count_path(path: &Path, uses: &mut [usize]) {
            uses[path.from] += 1;
            for hop in &path.hops {
                uses[hop.to] += 1;
            }
        }
        fn count_condition(condition: &Condition, uses: &mut [usize]) {
            // A read names the variable, but is no use in a pattern.
            let read = READ;
            match condition {
                Condition::SameNode(a, b) | Condition::SameValue(a, b) => {
                    uses[*a] += read;
                    uses[*b] += read;
                }
                Condition::Value(a, ..)
                | Condition::Key(a, _)
                | Condition::NodeId(a, _)
                | Condition::Raises(a) => {
                    uses[*a] += read;
                }
                Condition::Weight(..) => {}
                Condition::Or(left, right) => {
                    count_condition(left, uses);
                    count_condition(right, uses);
                }
                Condition::Not(inner) => count_condition(inner, uses),
                Condition::Exists(paths) => {
                    for path in paths {
                        count_path(path, uses);
                    }
                }
            }
        }
        for clause in &self.clauses {
            for path in &clause.paths {
                count_path(path, &mut uses);
            }
            for condition in &clause.conditions {
                count_condition(condition, &mut uses);
            }
        }
        for column in self.columns() {
            if let Column::Key(node) | Column::Value(node) = column {
                uses[node] += READ;
            }
        }
        if language == Language::Sql {
            uses[0] += READ;
        }
        uses
    }

    fn path_text(
        &self,
        path: &Path,
        language: Language,
        text: &mut Text,
        state: &mut (Vec<usize>, Vec<usize>),
    ) -> String {
        let mut out = self.node_text(path.from, language, text, state);
        for hop in &path.hops {
            let mut inner = String::new();
            if let Some(edge) = hop.edge {
                let _ = write!(inner, "e{edge}");
            }
            if !hop.kinds.is_empty() {
                let _ = write!(inner, ":{}", hop.kinds.join("|"));
            }
            if let Some(weight) = hop.weight {
                let constant = text.constant(language, weight);
                let _ = write!(inner, " {{w: {constant}}}");
            }
            let mut after = String::new();
            if let Some((least, most)) = hop.repeat {
                if language == Language::Cypher {
                    let _ = write!(inner, "*{least}..{most}");
                } else {
                    let _ = write!(after, "{{{least},{most}}}");
                }
            }
            let _ = match hop.way {
                Way::Out => write!(out, "-[{inner}]->{after}"),
                Way::In => write!(out, "<-[{inner}]-{after}"),
                Way::Either => write!(out, "-[{inner}]-{after}"),
            };
            out.push_str(&self.node_text(hop.to, language, text, state));
        }
        out
    }

    fn node_text(
        &self,
        node: usize,
        language: Language,
        text: &mut Text,
        state: &mut (Vec<usize>, Vec<usize>),
    ) -> String {
        let (total, seen) = state;
        let variable = &self.variables[node];
        let this_use = seen[node];
        seen[node] += 1;
        let first = this_use == 0;
        let last = this_use + 1 == total[node] % READ;
        let mut out = String::from("(");
        if !(variable.anonymous && total[node] == 1) {
            let _ = write!(out, "n{node}");
        }
        if let Some(label) = variable.label
            && (if variable.label_last { last } else { first })
        {
            let _ = write!(out, ":{label}");
        }
        if first {
            let mut properties = Vec::new();
            if let Some(key) = variable.key {
                properties.push(format!("k: {}", text.constant(language, key)));
            }
            if let Some(value) = variable.value {
                properties.push(format!("v: {}", text.constant(language, value)));
            }
            if !properties.is_empty() {
                let _ = write!(out, " {{{}}}", properties.join(", "));
            }
        }
        out.push(')');
        out
    }

    fn condition_text(
        &self,
        condition: &Condition,
        language: Language,
        ids: &HashMap<i64, u64>,
        text: &mut Text,
        state: &mut (Vec<usize>, Vec<usize>),
    ) -> String {
        match condition {
            Condition::SameNode(a, b) => format!("n{a} = n{b}"),
            Condition::Value(a, compare, value) => {
                let constant = text.constant(language, *value);
                format!("n{a}.v {} {constant}", compare.text())
            }
            Condition::Key(a, key) => {
                let constant = text.constant(language, *key);
                format!("n{a}.k = {constant}")
            }
            Condition::NodeId(a, key) => {
                // A node that is gone has no id: one that nothing has.
                let id = ids.get(key).map_or(1 << 40, |id| *id as i64);
                let constant = text.constant(language, id);
                format!("id(n{a}) = {constant}")
            }
            Condition::Weight(e, compare, weight) => {
                let constant = text.constant(language, *weight);
                format!("e{e}.w {} {constant}", compare.text())
            }
            Condition::SameValue(a, b) => format!("n{a}.v = n{b}.v"),
            Condition::Raises(a) => format!("100 / n{a}.v > 0"),
            Condition::Or(left, right) => format!(
                "({} OR {})",
                self.condition_text(left, language, ids, text, state),
                self.condition_text(right, language, ids, text, state)
            ),
            Condition::Not(inner) => format!(
                "NOT ({})",
                self.condition_text(inner, language, ids, text, state)
            ),
            Condition::Exists(paths) => {
                let inner: Vec<String> = paths
                    .iter()
                    .map(|path| self.path_text(path, language, text, state))
                    .collect();
                format!("EXISTS {{ MATCH {} }}", inner.join(", "))
            }
        }
    }

    fn render(&self, language: Language, ids: &HashMap<i64, u64>) -> Text {
        let mut text = Text {
            query: String::new(),
            parameters: HashMap::new(),
        };
        let total = self.uses_of_nodes(language);
        let seen = vec![0; self.variables.len()];
        let mut state = (total, seen);
        let mut body = String::new();
        for clause in &self.clauses {
            if clause.optional {
                body.push_str("OPTIONAL ");
            }
            let paths: Vec<String> = clause
                .paths
                .iter()
                .map(|path| self.path_text(path, language, &mut text, &mut state))
                .collect();
            let _ = write!(body, "MATCH {}", paths.join(", "));
            if !clause.conditions.is_empty() {
                let conditions: Vec<String> = clause
                    .conditions
                    .iter()
                    .map(|c| self.condition_text(c, language, ids, &mut text, &mut state))
                    .collect();
                if clause.stacked {
                    let keyword = if language == Language::Cypher {
                        "WITH * WHERE"
                    } else {
                        "FILTER"
                    };
                    for condition in conditions {
                        let _ = write!(body, " {keyword} {condition}");
                    }
                } else {
                    let _ = write!(body, " WHERE {}", conditions.join(" AND "));
                }
            }
            body.push(' ');
        }
        let named = |columns: &[Column]| -> Vec<String> {
            columns
                .iter()
                .enumerate()
                .map(|(index, column)| format!("{} AS c{index}", column.text()))
                .collect()
        };
        let order = |columns: &[Column]| -> String {
            (0..columns.len())
                .map(|index| format!("c{index}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        text.query = if language == Language::Sql {
            let columns = self.columns();
            let inner = if columns.is_empty() {
                "n0.k AS c0".to_string()
            } else {
                named(&columns).join(", ")
            };
            let table = format!("FROM GRAPH_TABLE ({body}COLUMNS ({inner}))");
            match &self.answer {
                Answer::Rows(columns) => format!("SELECT {} {table}", order(columns)),
                Answer::Distinct(columns) => format!("SELECT DISTINCT {} {table}", order(columns)),
                Answer::Count => format!("SELECT COUNT(*) AS c {table}"),
                Answer::CountBy(_) => format!("SELECT c0, COUNT(*) AS c {table} GROUP BY c0"),
                Answer::First(columns, count) => {
                    let names = order(columns);
                    format!("SELECT {names} {table} ORDER BY {names} LIMIT {count}")
                }
            }
        } else {
            match &self.answer {
                Answer::Rows(columns) => format!("{body}RETURN {}", named(columns).join(", ")),
                Answer::Distinct(columns) => {
                    format!("{body}RETURN DISTINCT {}", named(columns).join(", "))
                }
                Answer::Count => format!("{body}RETURN count(*) AS c"),
                Answer::CountBy(column) => {
                    format!("{body}RETURN {} AS c0, count(*) AS c", column.text())
                }
                Answer::First(columns, count) => format!(
                    "{body}RETURN {} ORDER BY {} LIMIT {count}",
                    named(columns).join(", "),
                    order(columns)
                ),
            }
        };
        text
    }
}

// ---------------------------------------------------------------------------
// Random queries, shaped after what the rules look for
// ---------------------------------------------------------------------------

struct QueryBuilder<'a> {
    random: &'a mut Random,
    graph: &'a Graph,
    variables: Vec<NodeVariable>,
    edge_variables: usize,
}

impl QueryBuilder<'_> {
    fn node(&mut self) -> usize {
        let mut variable = NodeVariable::default();
        if self.random.chance(15) {
            variable.label = Some(self.random.pick(&LABELS));
        }
        if self.random.chance(5) {
            variable.value = Some(self.random.below(5) as i64);
        }
        if self.random.chance(6) && !self.graph.nodes.is_empty() {
            let node = &self.graph.nodes[self.random.below(self.graph.nodes.len())];
            variable.key = Some(node.key);
        }
        variable.anonymous = self.random.chance(40);
        self.variables.push(variable);
        self.variables.len() - 1
    }

    fn kinds(&mut self) -> Vec<&'static str> {
        match self.random.below(100) {
            0..=39 => Vec::new(),
            40..=69 => vec![self.random.pick(&KINDS)],
            70..=97 => {
                let first = self.random.below(KINDS.len());
                let second = (first + 1 + self.random.below(KINDS.len() - 1)) % KINDS.len();
                vec![KINDS[first], KINDS[second]]
            }
            _ => vec!["NO_SUCH_KIND"],
        }
    }

    fn way(&mut self) -> Way {
        match self.random.below(10) {
            0..=4 => Way::Out,
            5..=6 => Way::In,
            _ => Way::Either,
        }
    }

    fn hop(&mut self, to: usize) -> Hop {
        let edge = self.random.chance(45).then(|| {
            self.edge_variables += 1;
            self.edge_variables - 1
        });
        Hop {
            edge,
            way: self.way(),
            kinds: self.kinds(),
            weight: self.random.chance(5).then(|| self.random.below(4) as i64),
            repeat: None,
            to,
        }
    }

    /// `n0-n1-...-n0`, one hop closing it.
    fn cycle(&mut self, length: usize) -> Path {
        let from = self.node();
        let mut hops = Vec::new();
        for _ in 1..length {
            let to = self.node();
            hops.push(self.hop(to));
        }
        hops.push(self.hop(from));
        if self.random.chance(25) {
            self.variables[from].label_last = true;
        }
        Path { from, hops }
    }

    fn chain(&mut self, length: usize) -> (Path, usize) {
        let from = self.node();
        let mut hops = Vec::new();
        let mut last = from;
        for _ in 0..length {
            last = self.node();
            hops.push(self.hop(last));
        }
        (Path { from, hops }, last)
    }

    fn shape(&mut self) -> Vec<Clause> {
        let one = |paths: Vec<Path>| {
            vec![Clause {
                optional: false,
                paths,
                conditions: Vec::new(),
                stacked: false,
            }]
        };
        match self.random.below(100) {
            // A cycle in one path.
            0..=29 => {
                let length = self.random.range(1, 4);
                let path = self.cycle(length);
                one(vec![path])
            }
            // A chain.
            30..=34 => {
                let length = self.random.range(1, 4);
                let (path, _) = self.chain(length);
                one(vec![path])
            }
            // A chain with its far end pinned, and perhaps its start, by a key or an id.
            35..=39 => {
                let length = self.random.range(1, 3);
                let (path, last) = self.chain(length);
                let from = path.from;
                let mut clauses = one(vec![path]);
                for end in [last, from] {
                    if self.graph.nodes.is_empty() || (end == from && self.random.chance(40)) {
                        continue;
                    }
                    let key = self.graph.nodes[self.random.below(self.graph.nodes.len())].key;
                    match self.random.below(3) {
                        0 => self.variables[end].key = Some(key),
                        1 => clauses[0].conditions.push(Condition::Key(end, key)),
                        _ => clauses[0].conditions.push(Condition::NodeId(end, key)),
                    }
                }
                clauses
            }
            // A chain whose end a WHERE says is its start.
            40..=49 => {
                let length = self.random.range(1, 3);
                let (path, last) = self.chain(length);
                let from = path.from;
                let mut clauses = one(vec![path]);
                clauses[0].conditions.push(if self.random.chance(50) {
                    Condition::SameNode(last, from)
                } else {
                    Condition::SameNode(from, last)
                });
                clauses
            }
            // Comma patterns that share a start, an end, or nothing.
            50..=54 => {
                let (a, b, c) = (self.node(), self.node(), self.node());
                let (first, second) = (self.hop(b), self.hop(c));
                let paths = match self.random.below(3) {
                    0 => vec![
                        Path {
                            from: a,
                            hops: vec![first],
                        },
                        Path {
                            from: a,
                            hops: vec![second],
                        },
                    ],
                    1 => {
                        let mut to_b = second;
                        to_b.to = b;
                        vec![
                            Path {
                                from: a,
                                hops: vec![first],
                            },
                            Path {
                                from: c,
                                hops: vec![to_b],
                            },
                        ]
                    }
                    _ => {
                        let d = self.node();
                        let mut to_d = second;
                        to_d.to = d;
                        vec![
                            Path {
                                from: a,
                                hops: vec![first],
                            },
                            Path {
                                from: c,
                                hops: vec![to_d],
                            },
                        ]
                    }
                };
                one(paths)
            }
            // A cycle of comma patterns, one hop each.
            55..=59 => {
                let length = self.random.range(2, 4);
                let first = self.node();
                let mut from = first;
                let mut paths = Vec::new();
                for step in 0..length {
                    let to = if step + 1 == length {
                        first
                    } else {
                        self.node()
                    };
                    let hop = self.hop(to);
                    paths.push(Path {
                        from,
                        hops: vec![hop],
                    });
                    from = to;
                }
                one(paths)
            }
            // Two MATCHHes, the second coming back.
            60..=69 => {
                let hops = self.random.below(2) + 1;
                let (path, last) = self.chain(hops);
                let from = path.from;
                let hop = self.hop(from);
                vec![
                    Clause {
                        optional: false,
                        paths: vec![path],
                        conditions: Vec::new(),
                        stacked: false,
                    },
                    Clause {
                        optional: false,
                        paths: vec![Path {
                            from: last,
                            hops: vec![hop],
                        }],
                        conditions: Vec::new(),
                        stacked: false,
                    },
                ]
            }
            // A quantified hop, perhaps back to where it starts.
            70..=79 => {
                let from = self.node();
                let to = if self.random.chance(30) {
                    from
                } else {
                    self.node()
                };
                let mut hop = self.hop(to);
                let least = self.random.below(2);
                hop.repeat = Some((least, least + self.random.range(1, 2)));
                hop.edge = None;
                hop.weight = None;
                one(vec![Path {
                    from,
                    hops: vec![hop],
                }])
            }
            // Something, and an OPTIONAL cycle over it.
            80..=89 => {
                let from = self.node();
                let length = self.random.range(1, 3);
                let mut path = self.cycle(length);
                let old_start = path.from;
                for hop in &mut path.hops {
                    if hop.to == old_start {
                        hop.to = from;
                    }
                }
                path.from = from;
                vec![
                    Clause {
                        optional: false,
                        paths: vec![Path {
                            from,
                            hops: Vec::new(),
                        }],
                        conditions: Vec::new(),
                        stacked: false,
                    },
                    Clause {
                        optional: true,
                        paths: vec![path],
                        conditions: Vec::new(),
                        stacked: false,
                    },
                ]
            }
            // A cycle in a subquery.
            _ => {
                let from = self.node();
                let length = self.random.range(1, 3);
                let mut path = self.cycle(length);
                let old_start = path.from;
                for hop in &mut path.hops {
                    hop.edge = None;
                    if hop.to == old_start {
                        hop.to = from;
                    }
                }
                path.from = from;
                let mut clauses = one(vec![Path {
                    from,
                    hops: Vec::new(),
                }]);
                clauses[0].conditions.push(Condition::Exists(vec![path]));
                clauses
            }
        }
    }

    /// Node variables a clause binds that later conditions and columns may read.
    fn bound_nodes(clause: &Clause) -> Vec<usize> {
        let mut nodes = Vec::new();
        for path in &clause.paths {
            nodes.push(path.from);
            nodes.extend(path.hops.iter().map(|hop| hop.to));
        }
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    fn bound_edges(clause: &Clause) -> Vec<usize> {
        clause
            .paths
            .iter()
            .flat_map(|path| path.hops.iter().filter_map(|hop| hop.edge))
            .collect()
    }

    fn condition(&mut self, nodes: &[usize], edges: &[usize]) -> Condition {
        let node = self.random.pick(nodes);
        let other = self.random.pick(nodes);
        let compare = self.random.pick(&[
            Compare::Equal,
            Compare::Different,
            Compare::Less,
            Compare::AtLeast,
        ]);
        let value = self.random.below(5) as i64;
        match self.random.below(100) {
            0..=34 => Condition::Value(node, compare, value),
            35..=44 => Condition::SameValue(node, other),
            45..=54 => Condition::Not(Box::new(Condition::SameNode(node, other))),
            55..=64 => Condition::Or(
                Box::new(Condition::SameNode(node, other)),
                Box::new(Condition::Value(node, compare, value)),
            ),
            65..=79 if !self.graph.nodes.is_empty() => {
                let key = self.graph.nodes[self.random.below(self.graph.nodes.len())].key;
                if self.random.chance(50) {
                    Condition::Key(node, key)
                } else {
                    Condition::NodeId(node, key)
                }
            }
            80..=99 if !edges.is_empty() => {
                Condition::Weight(self.random.pick(edges), compare, value)
            }
            _ => Condition::Value(other, compare, value),
        }
    }

    fn build(mut self) -> Query {
        let mut clauses = self.shape();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let count = clauses.len();
        for (index, clause) in clauses.iter_mut().enumerate() {
            nodes.extend(Self::bound_nodes(clause));
            nodes.sort_unstable();
            nodes.dedup();
            edges.extend(Self::bound_edges(clause));
            // Only the last clause filters: an OPTIONAL's conditions are its own.
            if index + 1 == count && self.random.chance(30) {
                for _ in 0..self.random.range(1, 2) {
                    let condition = self.condition(&nodes, &edges);
                    clause.conditions.push(condition);
                }
            }
        }
        let has_optional = clauses.iter().any(|clause| clause.optional);
        let mut columns = Vec::new();
        for &node in &nodes {
            if self.random.chance(55) {
                columns.push(if self.random.chance(85) {
                    Column::Key(node)
                } else {
                    Column::Value(node)
                });
            }
        }
        for &edge in &edges {
            if self.random.chance(55) {
                columns.push(Column::EdgeKey(edge));
            }
        }
        if columns.is_empty() {
            columns.push(Column::Key(nodes[0]));
        }
        let keys_only: Vec<Column> = columns
            .iter()
            .copied()
            .filter(|column| !matches!(column, Column::Value(_)))
            .collect();
        let answer = match self.random.below(100) {
            0..=44 => Answer::Rows(columns),
            45..=59 => Answer::Distinct(columns),
            60..=79 => Answer::Count,
            80..=91 => Answer::CountBy(self.random.pick(&columns)),
            _ if keys_only.is_empty() || has_optional => Answer::Rows(columns),
            _ => Answer::First(keys_only, self.random.range(1, 5)),
        };
        Query {
            variables: self.variables,
            edge_variables: self.edge_variables,
            clauses,
            answer,
        }
    }
}

fn random_query(random: &mut Random, graph: &Graph) -> Query {
    QueryBuilder {
        random,
        graph,
        variables: Vec::new(),
        edge_variables: 0,
    }
    .build()
}

// ---------------------------------------------------------------------------
// The brute-force enumerator: no adjacency, no index, no engine
// ---------------------------------------------------------------------------

type Row = Vec<Option<i64>>;

#[derive(Clone)]
struct Binding {
    nodes: Vec<Option<i64>>,
    edges: Vec<Option<i64>>,
}

struct Enumerator<'a> {
    graph: &'a Graph,
    query: &'a Query,
    edges_may_repeat: bool,
    budget: usize,
}

/// Over the budget: the query is given up, not answered.
struct TooLarge;

impl Enumerator<'_> {
    fn spend(&mut self) -> Result<(), TooLarge> {
        if self.budget == 0 {
            return Err(TooLarge);
        }
        self.budget -= 1;
        Ok(())
    }

    fn node(&self, key: i64) -> Option<&Node> {
        self.graph.nodes.iter().find(|node| node.key == key)
    }

    /// The nodes a hop reaches from `from` over one edge, and the edge's index.
    fn steps(&self, from: i64, hop: &Hop) -> Vec<(i64, usize)> {
        self.graph
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| hop.kinds.is_empty() || hop.kinds.contains(&edge.kind))
            .filter(|(_, edge)| hop.weight.is_none() || edge.weight == hop.weight)
            .filter_map(|(index, edge)| match hop.way {
                Way::Out => (edge.source == from).then_some((edge.target, index)),
                Way::In => (edge.target == from).then_some((edge.source, index)),
                // A self-loop once.
                Way::Either => {
                    if edge.source == from {
                        Some((edge.target, index))
                    } else {
                        (edge.target == from).then_some((edge.source, index))
                    }
                }
            })
            .collect()
    }

    fn satisfies_pattern(&self, binding: &Binding) -> bool {
        self.query
            .variables
            .iter()
            .enumerate()
            .all(|(index, variable)| {
                let Some(key) = binding.nodes[index] else {
                    return true;
                };
                let Some(node) = self.node(key) else {
                    return false;
                };
                variable
                    .label
                    .is_none_or(|label| node.labels.contains(&label))
                    && variable.key.is_none_or(|wanted| node.key == wanted)
                    && (variable.value.is_none() || node.value == variable.value)
            })
    }

    /// Every way to match `paths` from `binding`, each given to `found`.
    fn paths(
        &mut self,
        paths: &[Path],
        binding: &mut Binding,
        used: &mut Vec<usize>,
        found: &mut dyn FnMut(&Binding) -> bool,
    ) -> Result<(), TooLarge> {
        let Some((path, rest)) = paths.split_first() else {
            if self.satisfies_pattern(binding) {
                found(binding);
            }
            return Ok(());
        };
        if let Some(key) = binding.nodes[path.from] {
            return self.hops(&path.hops, key, rest, binding, used, found);
        }
        let keys: Vec<i64> = self.graph.nodes.iter().map(|node| node.key).collect();
        for key in keys {
            self.spend()?;
            binding.nodes[path.from] = Some(key);
            self.hops(&path.hops, key, rest, binding, used, found)?;
        }
        binding.nodes[path.from] = None;
        Ok(())
    }

    fn hops(
        &mut self,
        hops: &[Hop],
        at: i64,
        rest: &[Path],
        binding: &mut Binding,
        used: &mut Vec<usize>,
        found: &mut dyn FnMut(&Binding) -> bool,
    ) -> Result<(), TooLarge> {
        let Some((hop, later)) = hops.split_first() else {
            return self.paths(rest, binding, used, found);
        };
        match hop.repeat {
            None => {
                for (other, edge) in self.steps(at, hop) {
                    self.spend()?;
                    if !self.edges_may_repeat && used.contains(&edge) {
                        continue;
                    }
                    used.push(edge);
                    if let Some(variable) = hop.edge {
                        binding.edges[variable] = Some(self.graph.edges[edge].key);
                    }
                    self.arrive(hop, later, other, rest, binding, used, found)?;
                    if let Some(variable) = hop.edge {
                        binding.edges[variable] = None;
                    }
                    used.pop();
                }
                Ok(())
            }
            Some((least, most)) => {
                self.repeat(hop, later, at, 0, least, most, rest, binding, used, found)
            }
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the walk's whole state, one recursion"
    )]
    fn repeat(
        &mut self,
        hop: &Hop,
        later: &[Hop],
        at: i64,
        done: usize,
        least: usize,
        most: usize,
        rest: &[Path],
        binding: &mut Binding,
        used: &mut Vec<usize>,
        found: &mut dyn FnMut(&Binding) -> bool,
    ) -> Result<(), TooLarge> {
        self.spend()?;
        if done >= least {
            self.arrive(hop, later, at, rest, binding, used, found)?;
        }
        if done < most {
            for (other, edge) in self.steps(at, hop) {
                if !self.edges_may_repeat && used.contains(&edge) {
                    continue;
                }
                used.push(edge);
                self.repeat(
                    hop,
                    later,
                    other,
                    done + 1,
                    least,
                    most,
                    rest,
                    binding,
                    used,
                    found,
                )?;
                used.pop();
            }
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the walk's whole state, one recursion"
    )]
    fn arrive(
        &mut self,
        hop: &Hop,
        later: &[Hop],
        other: i64,
        rest: &[Path],
        binding: &mut Binding,
        used: &mut Vec<usize>,
        found: &mut dyn FnMut(&Binding) -> bool,
    ) -> Result<(), TooLarge> {
        match binding.nodes[hop.to] {
            Some(bound) if bound != other => Ok(()),
            Some(_) => self.hops(later, other, rest, binding, used, found),
            None => {
                binding.nodes[hop.to] = Some(other);
                let walked = self.hops(later, other, rest, binding, used, found);
                binding.nodes[hop.to] = None;
                walked
            }
        }
    }

    /// True, false or unknown; a null comparison is unknown.
    fn condition(
        &mut self,
        condition: &Condition,
        binding: &Binding,
    ) -> Result<Option<bool>, TooLarge> {
        let value = |this: &Self, node: usize| {
            binding.nodes[node]
                .and_then(|key| this.node(key))
                .and_then(|n| n.value)
        };
        Ok(match condition {
            Condition::SameNode(a, b) => match (binding.nodes[*a], binding.nodes[*b]) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            },
            Condition::Value(a, compare, wanted) => {
                value(self, *a).map(|got| compare.holds(got, *wanted))
            }
            Condition::Key(a, key) | Condition::NodeId(a, key) => {
                binding.nodes[*a].map(|got| got == *key)
            }
            Condition::Weight(e, compare, wanted) => binding.edges[*e]
                .and_then(|key| self.graph.edges.iter().find(|edge| edge.key == key))
                .and_then(|edge| edge.weight)
                .map(|got| compare.holds(got, *wanted)),
            Condition::SameValue(a, b) => match (value(self, *a), value(self, *b)) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            },
            Condition::Or(left, right) => {
                match (
                    self.condition(left, binding)?,
                    self.condition(right, binding)?,
                ) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            Condition::Not(inner) => self.condition(inner, binding)?.map(|got| !got),
            // A plan that succeeds did not divide by this 0, or another conjunct had
            // rejected the row: either way the row is not in its answer.
            Condition::Raises(a) => value(self, *a).map(|got| got > 0),
            Condition::Exists(paths) => {
                let mut any = false;
                let mut start = binding.clone();
                let mut used = Vec::new();
                self.paths(paths, &mut start, &mut used, &mut |_| {
                    any = true;
                    true
                })?;
                Some(any)
            }
        })
    }

    fn clause(&mut self, clause: &Clause, before: Vec<Binding>) -> Result<Vec<Binding>, TooLarge> {
        let mut after = Vec::new();
        for binding in before {
            let mut matched = Vec::new();
            let mut start = binding.clone();
            let mut used = Vec::new();
            self.paths(&clause.paths, &mut start, &mut used, &mut |found| {
                matched.push(found.clone());
                true
            })?;
            let mut kept = 0;
            for candidate in matched {
                let mut keep = true;
                for condition in &clause.conditions {
                    if self.condition(condition, &candidate)? != Some(true) {
                        keep = false;
                    }
                }
                if keep {
                    after.push(candidate);
                    kept += 1;
                    if after.len() > ANSWER_BUDGET {
                        return Err(TooLarge);
                    }
                }
            }
            if clause.optional && kept == 0 {
                after.push(binding);
            }
        }
        Ok(after)
    }

    fn answer(&mut self) -> Result<Vec<Row>, TooLarge> {
        let mut bindings = vec![Binding {
            nodes: vec![None; self.query.variables.len()],
            edges: vec![None; self.query.edge_variables],
        }];
        for clause in &self.query.clauses {
            bindings = self.clause(clause, bindings)?;
        }
        let read = |this: &Self, binding: &Binding, column: Column| match column {
            Column::Key(node) => binding.nodes[node],
            Column::Value(node) => binding.nodes[node]
                .and_then(|key| this.node(key))
                .and_then(|node| node.value),
            Column::EdgeKey(edge) => binding.edges[edge],
        };
        let row = |this: &Self, binding: &Binding, columns: &[Column]| -> Row {
            columns
                .iter()
                .map(|column| read(this, binding, *column))
                .collect()
        };
        let mut rows: Vec<Row> = match &self.query.answer {
            Answer::Rows(columns) | Answer::First(columns, _) => {
                bindings.iter().map(|b| row(self, b, columns)).collect()
            }
            Answer::Distinct(columns) => {
                let mut rows: Vec<Row> = bindings.iter().map(|b| row(self, b, columns)).collect();
                rows.sort();
                rows.dedup();
                rows
            }
            Answer::Count => vec![vec![Some(bindings.len() as i64)]],
            Answer::CountBy(column) => {
                let mut groups: BTreeMap<Option<i64>, i64> = BTreeMap::new();
                for binding in &bindings {
                    *groups.entry(read(self, binding, *column)).or_default() += 1;
                }
                groups
                    .into_iter()
                    .map(|(group, count)| vec![group, Some(count)])
                    .collect()
            }
        };
        rows.sort();
        if let Answer::First(_, count) = &self.query.answer {
            rows.truncate(*count);
        }
        Ok(rows)
    }
}

/// What the query answers on `graph`, sorted; `None` when it is too large to enumerate.
fn enumerate(graph: &Graph, query: &Query, language: Language) -> Option<Vec<Row>> {
    Enumerator {
        graph,
        query,
        edges_may_repeat: language.edges_may_repeat(),
        budget: ENUMERATION_BUDGET,
    }
    .answer()
    .ok()
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Setup {
    backward_edges: bool,
    indexed_key: bool,
    indexed_value: bool,
}

/// Where the graph stands when the queries run.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Committed,
    OwnWrites,
    AnotherOpenTransaction,
    AfterRollback,
    PastEpoch,
    AfterWrites,
    Reopened,
}

impl Stage {
    const ALL: [Self; 7] = [
        Self::Committed,
        Self::OwnWrites,
        Self::AnotherOpenTransaction,
        Self::AfterRollback,
        Self::PastEpoch,
        Self::AfterWrites,
        Self::Reopened,
    ];
}

fn config(setup: Setup, base: Config) -> Config {
    let config = base.with_history_retention(HistoryRetention::Forever);
    if setup.backward_edges {
        config
    } else {
        config.without_backward_edges()
    }
}

fn load(db: &GrafeoDB, setup: Setup, graph: &Graph) {
    if setup.indexed_key {
        db.create_property_index("k").unwrap();
    }
    if setup.indexed_value {
        db.create_property_index("v").unwrap();
    }
    let mut ids = HashMap::new();
    for node in &graph.nodes {
        let mut properties = vec![("k", Value::Int64(node.key))];
        if let Some(value) = node.value {
            properties.push(("v", Value::Int64(value)));
        }
        let id = db.create_node_with_props(&node.labels, properties).unwrap();
        ids.insert(node.key, id);
    }
    for edge in &graph.edges {
        let mut properties = vec![("k", Value::Int64(edge.key))];
        if let Some(weight) = edge.weight {
            properties.push(("w", Value::Int64(weight)));
        }
        db.create_edge_with_props(ids[&edge.source], ids[&edge.target], edge.kind, properties)
            .unwrap();
    }
}

fn apply(session: &Session, writes: &[Write]) {
    for write in writes {
        session
            .execute(&write.text())
            .unwrap_or_else(|error| panic!("{}: {error}", write.text()));
    }
}

fn node_ids(session: &Session) -> HashMap<i64, u64> {
    session
        .execute("MATCH (n) RETURN n.k, id(n)")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| match (&row[0], &row[1]) {
            (Value::Int64(key), Value::Int64(id)) => Some((*key, *id as u64)),
            _ => None,
        })
        .collect()
}

/// Runs `body` with a session at `stage`, and the graph it should see there.
fn at_stage(
    setup: Setup,
    graph: &Graph,
    writes: &[Write],
    stage: Stage,
    body: &mut dyn FnMut(&mut Session, &HashMap<i64, u64>, &Graph),
) {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("graph.grafeo");
    let after = graph.after(writes);
    if stage == Stage::Reopened {
        {
            let db = GrafeoDB::with_config(config(setup, Config::persistent(&file))).unwrap();
            load(&db, setup, graph);
            apply(&db.session(), writes);
            db.close().unwrap();
        }
        let db = GrafeoDB::with_config(config(setup, Config::persistent(&file))).unwrap();
        let mut session = db.session();
        let ids = node_ids(&session);
        body(&mut session, &ids, &after);
        db.close().unwrap();
        return;
    }
    let db = GrafeoDB::with_config(config(setup, Config::in_memory())).unwrap();
    load(&db, setup, graph);
    let mut session = db.session();
    match stage {
        Stage::Committed => {
            let ids = node_ids(&session);
            body(&mut session, &ids, graph);
        }
        Stage::OwnWrites => {
            session.begin_transaction().unwrap();
            apply(&session, writes);
            let ids = node_ids(&session);
            body(&mut session, &ids, &after);
            session.rollback().unwrap();
        }
        Stage::AnotherOpenTransaction => {
            let mut other = db.session();
            other.begin_transaction().unwrap();
            apply(&other, writes);
            let ids = node_ids(&session);
            body(&mut session, &ids, graph);
            other.rollback().unwrap();
        }
        Stage::AfterRollback => {
            session.begin_transaction().unwrap();
            apply(&session, writes);
            session.rollback().unwrap();
            let ids = node_ids(&session);
            body(&mut session, &ids, graph);
        }
        Stage::PastEpoch => {
            let ids = node_ids(&session);
            let epoch: EpochId = db.current_epoch();
            apply(&session, writes);
            session.set_viewing_epoch(epoch);
            body(&mut session, &ids, graph);
            session.clear_viewing_epoch();
        }
        Stage::AfterWrites => {
            apply(&session, writes);
            let ids = node_ids(&session);
            body(&mut session, &ids, &after);
        }
        Stage::Reopened => unreachable!(),
    }
}

fn run(session: &Session, language: Language, text: &Text) -> Result<Vec<Row>, String> {
    let result = match language {
        Language::Gql => session.execute(&text.query),
        Language::GqlParameters => {
            session.execute_with_params(&text.query, text.parameters.clone())
        }
        Language::Cypher => session.execute_cypher(&text.query),
        Language::Sql => session.execute_sql(&text.query),
    }
    .map_err(|error| error.to_string())?;
    result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::Int64(number) => Ok(Some(*number)),
                    Value::Null => Ok(None),
                    other => Err(format!("a value this test never asks for: {other:?}")),
                })
                .collect()
        })
        .collect()
}

fn sorted_unless_ordered(mut rows: Vec<Row>, query: &Query) -> Vec<Row> {
    if !matches!(query.answer, Answer::First(..)) {
        rows.sort();
    }
    rows
}

/// The rules that choose a plan's shape, each of which must work alone too; the others say
/// whether one way into the store is used, and go off alone only.
const PLAN_RULES: [PlannerRule; 8] = [
    PlannerRule::JoinOrder,
    PlannerRule::SoughtEnd,
    PlannerRule::FactorizedExecution,
    PlannerRule::LabelCount,
    PlannerRule::Reachability,
    PlannerRule::ExpandInto,
    PlannerRule::PinnedFarEnd,
    PlannerRule::IntersectExpand,
];

/// The rule sets each stage compares: all of them where it costs little.
fn rule_sets(stage: Stage) -> Vec<(String, PlannerRules)> {
    let mut sets = vec![
        ("all rules".to_string(), PlannerRules::all()),
        ("no rule".to_string(), PlannerRules::plain()),
    ];
    if stage == Stage::Committed {
        for rule in PlannerRule::ALL {
            sets.push((
                format!("all but {rule:?}"),
                PlannerRules::all().with(rule, false),
            ));
        }
        for rule in PLAN_RULES {
            sets.push((
                format!("only {rule:?}"),
                PlannerRules::plain().with(rule, true),
            ));
        }
    } else {
        for rule in [
            PlannerRule::ExpandInto,
            PlannerRule::PinnedFarEnd,
            PlannerRule::IntersectExpand,
        ] {
            sets.push((
                format!("all but {rule:?}"),
                PlannerRules::all().with(rule, false),
            ));
        }
    }
    sets
}

// ---------------------------------------------------------------------------
// A difference, shrunk
// ---------------------------------------------------------------------------

struct Scenario<'a> {
    seed: u64,
    setup: Setup,
    graph: Graph,
    writes: &'a [Write],
    stage: Stage,
    query: &'a Query,
    language: Language,
    rules: PlannerRules,
    rules_name: String,
}

impl Scenario<'_> {
    /// What the enumerator and the engine answer, if they differ.
    fn difference(&self, graph: &Graph) -> Option<(Vec<Row>, Result<Vec<Row>, String>, String)> {
        let mut found = None;
        at_stage(
            self.setup,
            graph,
            self.writes,
            self.stage,
            &mut |session, ids, seen| {
                session.set_planner_rules(self.rules);
                let Some(expected) = enumerate(seen, self.query, self.language) else {
                    return;
                };
                let text = self.query.render(self.language, ids);
                let got = run(session, self.language, &text)
                    .map(|rows| sorted_unless_ordered(rows, self.query));
                if got.as_ref() != Ok(&expected) {
                    found = Some((expected, got, text.query));
                }
            },
        );
        found
    }

    /// The smallest graph this finds that still differs.
    fn shrunk(&self) -> Graph {
        let mut graph = self.graph.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for index in (0..graph.edges.len()).rev() {
                let mut smaller = graph.clone();
                smaller.edges.remove(index);
                if self.difference(&smaller).is_some() {
                    graph = smaller;
                    changed = true;
                }
            }
            for index in (0..graph.nodes.len()).rev() {
                let key = graph.nodes[index].key;
                let mut smaller = graph.clone();
                smaller.nodes.remove(index);
                smaller.edges.retain(|e| e.source != key && e.target != key);
                if self.difference(&smaller).is_some() {
                    graph = smaller;
                    changed = true;
                }
            }
        }
        graph
    }

    fn report(&self) -> String {
        let graph = self.shrunk();
        let mut out = format!(
            "seed {} ({:?}) stage {:?} {:?}, with {}\n",
            self.seed, self.setup, self.stage, self.language, self.rules_name
        );
        match self.difference(&graph) {
            Some((expected, got, text)) => {
                let _ = writeln!(out, "  query:    {text}");
                let _ = writeln!(out, "  expected: {expected:?}");
                let _ = writeln!(out, "  got:      {got:?}");
            }
            None => out.push_str("  (no longer differs on the shrunken graph)\n"),
        }
        let _ = writeln!(
            out,
            "  graph, {} nodes and {} edges:",
            graph.nodes.len(),
            graph.edges.len()
        );
        for line in graph.statements().lines() {
            let _ = writeln!(out, "    {line}");
        }
        if !self.writes.is_empty() && self.stage != Stage::Committed {
            let _ = writeln!(out, "  writes:");
            for write in self.writes {
                let _ = writeln!(out, "    {}", write.text());
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Tally {
    asked: usize,
    given_up: usize,
    with_rows: usize,
    closing_hops: usize,
    differences: Vec<String>,
    /// More differences than are reported, by what they are.
    other_differences: usize,
}

struct Plan<'a> {
    seed: u64,
    setup: Setup,
    graph: &'a Graph,
    writes: &'a [Write],
    queries: &'a [Query],
}

fn check_stage(plan: &Plan<'_>, stage: Stage, sets: &[(String, PlannerRules)], tally: &mut Tally) {
    let mut failing: Vec<Scenario<'_>> = Vec::new();
    at_stage(
        plan.setup,
        plan.graph,
        plan.writes,
        stage,
        &mut |session, ids, seen| {
            for query in plan.queries {
                for language in Language::ALL {
                    if language == Language::Sql && !query.fits_sql() {
                        continue;
                    }
                    let Some(expected) = enumerate(seen, query, language) else {
                        tally.given_up += 1;
                        continue;
                    };
                    tally.asked += 1;
                    if expected.iter().any(|row| row != &vec![Some(0)]) {
                        tally.with_rows += 1;
                    }
                    let text = query.render(language, ids);

                    for (name, rules) in sets {
                        session.set_planner_rules(*rules);
                        let got = run(session, language, &text);

                        // Twice: the second reads the plan cache.
                        let again = run(session, language, &text);
                        let sorted = got.clone().map(|rows| sorted_unless_ordered(rows, query));
                        if sorted.as_ref() != Ok(&expected) || again != got {
                            failing.push(Scenario {
                                seed: plan.seed,
                                setup: plan.setup,
                                graph: plan.graph.clone(),
                                writes: plan.writes,
                                stage,
                                query,
                                language,
                                rules: *rules,
                                rules_name: name.clone(),
                            });
                        }
                    }
                    // ExpandInto keeps the order the list gives, row for row (the intersection's
                    // order is its own, so both sides run without it).
                    let without = PlannerRules::all().with(PlannerRule::IntersectExpand, false);
                    session.set_planner_rules(without.with(PlannerRule::ExpandInto, false));
                    if let Ok(listed) = run(session, language, &text) {
                        session.set_planner_rules(without);
                        if let Ok(looked_up) = run(session, language, &text)
                            && looked_up != listed
                            && !matches!(query.answer, Answer::Distinct(_) | Answer::CountBy(_))
                        {
                            tally.differences.push(format!(
                            "seed {} stage {stage:?} {language:?}: ExpandInto changes the row order\n  query: {}\n  looked up: {looked_up:?}\n  listed:    {listed:?}\n",
                            plan.seed, text.query
                        ));
                        }
                    }
                    session.set_planner_rules(PlannerRules::all());
                    if let Ok(result) = session.execute(&format!(
                        "EXPLAIN {}",
                        query.render(Language::Gql, ids).query
                    )) && format!("{:?}", result.rows()).contains("expand into")
                    {
                        tally.closing_hops += 1;
                    }
                }
            }
        },
    );
    for scenario in failing {
        if tally.differences.len() < REPORTED {
            tally.differences.push(scenario.report());
        } else {
            tally.other_differences += 1;
        }
    }
}

fn check_seed(seed: u64, tally: &mut Tally) {
    let mut random = Random::new(seed);
    let graph = Graph::random(&mut random, 12, 36);
    let writes = graph.random_writes(&mut random);
    let setup = Setup {
        backward_edges: !random.chance(25),
        indexed_key: random.chance(40),
        indexed_value: random.chance(20),
    };
    let queries: Vec<Query> = (0..QUERIES_PER_GRAPH)
        .map(|_| random_query(&mut random, &graph))
        .collect();
    let plan = Plan {
        seed,
        setup,
        graph: &graph,
        writes: &writes,
        queries: &queries,
    };
    for stage in Stage::ALL {
        check_stage(&plan, stage, &rule_sets(stage), tally);
    }
}

fn verdict(tally: &Tally) {
    eprintln!(
        "{} answers compared, {} with rows, {} over a closing hop, {} given up",
        tally.asked, tally.with_rows, tally.closing_hops, tally.given_up
    );
    assert!(
        tally.differences.is_empty(),
        "{} differences reported and {} more:\n\n{}",
        tally.differences.len(),
        tally.other_differences,
        tally.differences.join("\n")
    );
}

fn seeds() -> Vec<u64> {
    if let Some(seed) = std::env::var("GRAFEO_RULES_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return vec![seed];
    }
    let count = std::env::var("GRAFEO_RULES_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(FIXED_SEEDS);
    (0..count).collect()
}

/// Rules on, rules off and the enumerator agree, at every stage of the graph.
#[test]
fn every_rule_set_answers_what_the_enumerator_answers() {
    let mut tally = Tally::default();
    for seed in seeds() {
        check_seed(seed, &mut tally);
        if tally.differences.len() >= REPORTED {
            break;
        }
    }
    let enough = seeds().len() >= FIXED_SEEDS as usize;
    if enough && tally.differences.is_empty() {
        // The run proves little unless the rows and the shapes were there.
        assert!(tally.asked > 1_000, "only {} answers", tally.asked);
        assert!(
            tally.with_rows * 5 > tally.asked * 2,
            "only {} of {} had rows",
            tally.with_rows,
            tally.asked
        );
        assert!(
            tally.closing_hops > 100,
            "only {} closing hops",
            tally.closing_hops
        );
        assert!(
            tally.given_up * 10 < tally.asked,
            "gave up on {}",
            tally.given_up
        );
    }
    verdict(&tally);
}

/// Enough edges that the adjacency merges its delta into the sorted list, with deltas and
/// removed edges after it, in memory and read back from a file.
#[test]
fn rule_sets_agree_over_merged_adjacency() {
    let mut tally = Tally::default();
    for seed in [101_u64, 102, 103] {
        let mut random = Random::new(seed);
        let mut graph = Graph::random(&mut random, 12, 20);
        for key in graph.nodes.len()..400 {
            graph.nodes.push(Node {
                key: key as i64,
                labels: vec![random.pick(&LABELS)],
                value: Some(random.below(5) as i64),
            });
        }
        // More than a merge takes: a thin graph, and a dense core where the cycles are.
        for key in graph.edges.len()..1_500 {
            let core = key % 12 == 0;
            let within = if core { 12 } else { 400 };
            graph.edges.push(Edge {
                key: key as i64,
                source: random.below(within) as i64,
                target: random.below(within) as i64,
                kind: random.pick(&KINDS),
                weight: random.chance(70).then(|| random.below(4) as i64),
            });
        }
        let writes = graph.random_writes(&mut random);
        let setup = Setup {
            backward_edges: seed != 103,
            indexed_key: false,
            indexed_value: false,
        };
        let queries: Vec<Query> = (0..8).map(|_| random_query(&mut random, &graph)).collect();
        let plan = Plan {
            seed,
            setup,
            graph: &graph,
            writes: &writes,
            queries: &queries,
        };
        let sets = rule_sets(Stage::Reopened);
        for stage in [Stage::Committed, Stage::OwnWrites, Stage::Reopened] {
            check_stage(&plan, stage, &sets, &mut tally);
        }
    }
    verdict(&tally);
}

// ---------------------------------------------------------------------------
// Conjuncts that raise: the contract, rule by rule
// ---------------------------------------------------------------------------

/// `n0-...-n<last>` closed by a condition, with `100 / n.v > 0` and a guard around it in
/// every written order, in one `WHERE` and as statements of their own.
fn raising_query(random: &mut Random, graph: &Graph) -> Query {
    let mut builder = QueryBuilder {
        random,
        graph,
        variables: Vec::new(),
        edge_variables: 0,
    };
    let length = builder.random.range(1, 3);
    let (path, last) = builder.chain(length);
    let from = path.from;
    let mut nodes = vec![from];
    nodes.extend(path.hops.iter().map(|hop| hop.to));
    let guarded = builder.random.pick(&nodes);
    let raises = Condition::Raises(guarded);
    let guard = Condition::Value(guarded, Compare::Different, 0);
    let closing = Condition::SameNode(last, from);
    let mut conditions = match builder.random.below(6) {
        0 => vec![guard, closing, raises],
        1 => vec![guard, raises, closing],
        2 => vec![closing, guard, raises],
        3 => vec![closing, raises, guard],
        4 => vec![raises, guard, closing],
        _ => vec![raises, closing, guard],
    };
    if builder.random.chance(30) {
        conditions.remove(builder.random.below(3));
    }
    let stacked = builder.random.chance(60);
    let mut paths = vec![path];
    // A second part through one of its nodes, in comma form: the join search moves this one.
    if builder.random.chance(50) {
        let shared = builder.random.pick(&nodes);
        let far = builder.node();
        let hop = builder.hop(far);
        paths.push(Path {
            from: shared,
            hops: vec![hop],
        });
    }
    Query {
        variables: builder.variables,
        edge_variables: builder.edge_variables,
        clauses: vec![Clause {
            optional: false,
            paths,
            conditions,
            stacked,
        }],
        answer: Answer::Rows(vec![Column::Key(from), Column::Key(last)]),
    }
}

/// The one exception to the contract, decided by the user: with another part of the pattern
/// empty, the plan without the rule never reads the part with the raising filter (a hash join
/// with an empty build side skips its other side) and returns no rows, where a plan with the
/// rule may run that filter first and raise. Only for no rows, and only with an empty part.
fn the_empty_part_exception(
    graph: &Graph,
    query: &Query,
    language: Language,
    without_rule: &[Row],
) -> bool {
    if !without_rule.is_empty() {
        return false;
    }
    query.clauses[0].paths.len() > 1
        && query.clauses[0].paths.iter().any(|path| {
            let part = Query {
                variables: query.variables.clone(),
                edge_variables: query.edge_variables,
                clauses: vec![Clause {
                    optional: false,
                    paths: vec![path.clone()],
                    conditions: Vec::new(),
                    stacked: false,
                }],
                answer: Answer::Rows(vec![Column::Key(path.from)]),
            };
            enumerate(graph, &part, language).is_some_and(|rows| rows.is_empty())
        })
}

/// The pairs the contract compares: each rule on and off, among all the others and alone.
fn contract_pairs() -> Vec<(String, PlannerRules, PlannerRules)> {
    let mut pairs = vec![(
        "all rules against no rule".to_string(),
        PlannerRules::all(),
        PlannerRules::plain(),
    )];
    for rule in PlannerRule::ALL {
        pairs.push((
            format!("{rule:?} among the others"),
            PlannerRules::all(),
            PlannerRules::all().with(rule, false),
        ));
    }
    for rule in PLAN_RULES {
        pairs.push((
            format!("{rule:?} alone"),
            PlannerRules::plain().with(rule, true),
            PlannerRules::plain(),
        ));
    }
    pairs
}

/// The plan with rules off is the reference, and it moved in 0.6.2: pushdown no longer copies a
/// raising filter to both sides of a join. It goes to the side that holds every written check on
/// its variables, or stays above the join when they are split, so `FILTER a.k <> 0 FILTER
/// 100 / a.k > 0` and `(a:A {k: 3}) ... WHERE 100 / a.k > 0` no longer raise with rules off.
///
/// With a conjunct that raises: both succeed with the same rows, or rule off fails (allowed),
/// and whichever plan succeeds answers what the enumerator answers. Rule off succeeding
/// while rule on fails is the one thing a rule may not do, but for
/// [`the_empty_part_exception`].
#[test]
fn a_rule_never_fails_a_query_that_succeeds_without_it() {
    let mut differences = Vec::new();
    let mut both_succeed = 0;
    let mut rule_off_only_fails = 0;
    let mut empty_part_raises = 0;
    for seed in seeds() {
        let mut random = Random::new(seed ^ 0xC0DE);
        // Values of 0 are what raises.
        let mut graph = Graph::random(&mut random, 10, 30);
        for node in &mut graph.nodes {
            if node.value.is_some() && random.chance(30) {
                node.value = Some(0);
            }
        }
        let setup = Setup {
            backward_edges: !random.chance(25),
            indexed_key: false,
            indexed_value: random.chance(25),
        };
        let queries: Vec<Query> = (0..QUERIES_PER_GRAPH)
            .map(|_| raising_query(&mut random, &graph))
            .collect();
        at_stage(
            setup,
            &graph,
            &[],
            Stage::Committed,
            &mut |session, ids, seen| {
                for query in &queries {
                    for language in Language::ALL {
                        if language == Language::Sql && !query.fits_sql() {
                            continue;
                        }
                        let Some(expected) = enumerate(seen, query, language) else {
                            continue;
                        };
                        let text = query.render(language, ids);
                        for (name, on, off) in contract_pairs() {
                            session.set_planner_rules(on);
                            let with_rule = run(session, language, &text);
                            session.set_planner_rules(off);
                            let without_rule = run(session, language, &text);
                            match (&with_rule, &without_rule) {
                            (Ok(with_rule), Ok(without_rule)) => {
                                both_succeed += 1;
                                let sorted = sorted_unless_ordered(with_rule.clone(), query);
                                if sorted != sorted_unless_ordered(without_rule.clone(), query)
                                    || sorted != expected
                                {
                                    differences.push(format!(
                                        "seed {seed} {language:?}, {name}: rows differ\n  query: {}\n  rule on:  {with_rule:?}\n  rule off: {without_rule:?}\n  expected: {expected:?}\n{}",
                                        text.query,
                                        graph.statements()
                                    ));
                                }
                            }
                            (Ok(with_rule), Err(_)) => {
                                rule_off_only_fails += 1;
                                if sorted_unless_ordered(with_rule.clone(), query) != expected {
                                    differences.push(format!(
                                        "seed {seed} {language:?}, {name}: wrong rows where rule off fails\n  query: {}\n  rule on:  {with_rule:?}\n  expected: {expected:?}\n{}",
                                        text.query,
                                        graph.statements()
                                    ));
                                }
                            }
                            (Err(_), Ok(without_rule))
                                if the_empty_part_exception(
                                    seen,
                                    query,
                                    language,
                                    without_rule,
                                ) =>
                            {
                                empty_part_raises += 1;
                            }
                            (Err(error), Ok(_)) => differences.push(format!(
                                "seed {seed} {language:?}, {name}: the rule fails a query that succeeds without it\n  query: {}\n  error: {error}\n{}",
                                text.query,
                                graph.statements()
                            )),
                            (Err(_), Err(_)) => {}
                        }
                        }
                    }
                }
            },
        );
        if differences.len() >= REPORTED {
            break;
        }
    }
    eprintln!(
        "{both_succeed} pairs both succeed, {rule_off_only_fails} where only rule off fails, \
         {empty_part_raises} where an empty part lets a rule raise"
    );
    assert!(
        differences.is_empty(),
        "{} differences:\n\n{}",
        differences.len(),
        differences.join("\n")
    );
    if seeds().len() >= FIXED_SEEDS as usize {
        assert!(both_succeed > 500, "only {both_succeed} pairs succeeded");
    }
}

// ---------------------------------------------------------------------------
// An index finds what a scan finds
// ---------------------------------------------------------------------------

/// Stored values of many kinds, filters of every operator and kind: the same rows with no
/// index, a hash index and a btree index, also for a key changed and for a node removed, and
/// in a transaction that has not committed them. What the rows are is ISO GA04's to say.
#[test]
fn an_index_finds_what_a_scan_finds() {
    const STORED: [(&str, &str); 10] = [
        ("Alix", "5"),
        ("Gus", "'5'"),
        ("Vincent", "5.0"),
        ("Mia", "'five'"),
        ("Jules", "'05'"),
        ("Django", "-3"),
        ("Shosanna", "'z'"),
        ("Hans", "true"),
        ("Lucas", "2.5"),
        ("Marcus", "9007199254740993"),
    ];
    const KEYS: [&str; 10] = [
        "4",
        "5",
        "5.0",
        "2.5",
        "'4'",
        "'5'",
        "true",
        "null",
        "9007199254740992",
        "9007199254740992.0",
    ];
    let mut filters: Vec<String> = Vec::new();
    for operator in ["=", "<>", "<", "<=", ">", ">="] {
        for key in KEYS {
            filters.push(format!("n.id {operator} {key}"));
        }
    }
    for list in [
        "[5]",
        "['5']",
        "[5, '5']",
        "[5.0]",
        "[2.5, 'z', true]",
        "[null]",
    ] {
        filters.push(format!("n.id IN {list}"));
    }
    let build = |index: Option<&str>| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        if let Some(index) = index {
            session.execute(index).unwrap();
        }
        for (name, value) in STORED {
            session
                .execute(&format!("INSERT (:T {{name: '{name}', id: {value}}})"))
                .unwrap();
        }
        session.execute("INSERT (:T {name: 'Butch'})").unwrap();
        db
    };
    let dbs = [
        ("no index", build(None)),
        ("hash", build(Some("CREATE INDEX i FOR (n:T) ON (n.id)"))),
        (
            "btree",
            build(Some("CREATE INDEX i FOR (n:T) ON (n.id) USING BTREE")),
        ),
    ];
    let answer = |session: &Session, filter: &str| -> Result<Vec<String>, String> {
        let query = format!("MATCH (n:T) WHERE {filter} RETURN n.name");
        let result = session.execute(&query).map_err(|error| error.to_string())?;
        let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
        rows.sort();
        Ok(rows)
    };
    let mut differences = Vec::new();
    let mut compare = |stage: &str, sessions: &[(&str, &Session)]| {
        for filter in &filters {
            let scan = answer(sessions[0].1, filter);
            for (name, session) in &sessions[1..] {
                let got = answer(session, filter);
                if got != scan {
                    differences.push(format!(
                        "{stage}, {name}: {filter}\n  index {got:?}\n  scan  {scan:?}"
                    ));
                }
            }
        }
    };
    let mut sessions: Vec<(&str, Session)> =
        dbs.iter().map(|(name, db)| (*name, db.session())).collect();
    {
        let list: Vec<(&str, &Session)> = sessions.iter().map(|(n, s)| (*n, s)).collect();
        compare("committed", &list);
    }
    for (_, session) in &mut sessions {
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (n:T {name: 'Mia'}) SET n.id = 5")
            .unwrap();
        session
            .execute("MATCH (n:T {name: 'Alix'}) SET n.id = 'five'")
            .unwrap();
        session
            .execute("MATCH (n:T {name: 'Vincent'}) DETACH DELETE n")
            .unwrap();
        session
            .execute("INSERT (:T {name: 'Harm', id: 5})")
            .unwrap();
    }
    {
        let list: Vec<(&str, &Session)> = sessions.iter().map(|(n, s)| (*n, s)).collect();
        compare("own writes", &list);
    }
    for (_, session) in &mut sessions {
        session.rollback().unwrap();
    }
    {
        let list: Vec<(&str, &Session)> = sessions.iter().map(|(n, s)| (*n, s)).collect();
        compare("after the rollback", &list);
    }
    for (_, session) in &mut sessions {
        session
            .execute("MATCH (n:T {name: 'Mia'}) SET n.id = 5")
            .unwrap();
        session
            .execute("MATCH (n:T {name: 'Alix'}) SET n.id = 'five'")
            .unwrap();
        session
            .execute("MATCH (n:T {name: 'Vincent'}) DETACH DELETE n")
            .unwrap();
    }
    {
        let list: Vec<(&str, &Session)> = sessions.iter().map(|(n, s)| (*n, s)).collect();
        compare("after commits", &list);
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}
