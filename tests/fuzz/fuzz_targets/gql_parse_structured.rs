//! Structured fuzz target for the GQL parser.
//!
//! Generates syntactically plausible GQL queries from structured random input.
//! More effective than raw bytes because every input exercises meaningful
//! parser paths.
//!
//! Run: cargo +nightly fuzz run gql_parse_structured --fuzz-dir tests/fuzz

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

/// A randomly generated GQL query built from structured components.
#[derive(Debug, Arbitrary)]
struct FuzzQuery {
    clause: Clause,
    pattern: Pattern,
    filter: Option<Filter>,
    ret: Return,
    modifiers: Modifiers,
}

#[derive(Debug, Arbitrary)]
enum Clause {
    Match,
    OptionalMatch,
    Insert,
    Merge,
}

#[derive(Debug, Arbitrary)]
struct Pattern {
    node: NodePattern,
    edge: Option<EdgePattern>,
    second_node: Option<NodePattern>,
}

#[derive(Debug, Arbitrary)]
struct NodePattern {
    variable: Option<VarName>,
    label: Option<Label>,
    props: Vec<(PropName, PropValue)>,
}

#[derive(Debug, Arbitrary)]
struct EdgePattern {
    variable: Option<VarName>,
    edge_type: Option<Label>,
    direction: Direction,
}

#[derive(Debug, Arbitrary)]
enum Direction {
    Outgoing,
    Incoming,
    Undirected,
}

#[derive(Debug, Arbitrary)]
struct Filter {
    left: PropAccess,
    op: CompOp,
    right: PropValue,
}

#[derive(Debug, Arbitrary)]
struct Return {
    items: Vec<ReturnItem>,
    distinct: bool,
}

#[derive(Debug, Arbitrary)]
enum ReturnItem {
    Star,
    Property(PropAccess),
    Count(Option<PropAccess>),
    Alias(PropAccess, VarName),
}

#[derive(Debug, Arbitrary)]
struct Modifiers {
    order_by: Option<PropAccess>,
    order_desc: bool,
    limit: Option<u8>,
    skip: Option<u8>,
}

#[derive(Debug, Arbitrary)]
struct PropAccess {
    var: VarName,
    prop: PropName,
}

#[derive(Debug, Arbitrary)]
enum CompOp {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

#[derive(Debug, Arbitrary)]
enum PropValue {
    Int(i32),
    Float(f32),
    Str(SmallStr),
    Bool(bool),
    Null,
}

#[derive(Debug, Arbitrary)]
struct VarName(u8); // maps to a-z
#[derive(Debug, Arbitrary)]
struct Label(u8); // maps to Person, City, Company, etc.
#[derive(Debug, Arbitrary)]
struct PropName(u8); // maps to name, age, city, etc.
#[derive(Debug, Arbitrary)]
struct SmallStr(u8); // maps to short string values

// ---------------------------------------------------------------------------
// Rendering to GQL strings
// ---------------------------------------------------------------------------

const LABELS: &[&str] = &[
    "Person", "City", "Company", "Node", "Item", "Tag", "Sensor", "Data",
];
const PROPS: &[&str] = &[
    "name", "age", "city", "val", "score", "status", "role", "since",
];
const VARS: &[&str] = &["n", "m", "a", "b", "c", "x", "y", "p", "q", "r"];
const STRINGS: &[&str] = &[
    "Alix",
    "Gus",
    "Vincent",
    "Amsterdam",
    "Berlin",
    "test",
    "",
    "hello",
];

impl FuzzQuery {
    fn to_gql(&self) -> String {
        let mut q = String::new();

        // Clause
        match self.clause {
            Clause::Match => q.push_str("MATCH "),
            Clause::OptionalMatch => q.push_str("OPTIONAL MATCH "),
            Clause::Insert => q.push_str("INSERT "),
            Clause::Merge => q.push_str("MERGE "),
        }

        // Pattern
        q.push_str(&self.pattern.to_gql());

        // WHERE
        if let Some(f) = &self.filter {
            q.push_str(" WHERE ");
            q.push_str(&f.to_gql());
        }

        // For INSERT/MERGE, we don't need RETURN
        if matches!(self.clause, Clause::Match | Clause::OptionalMatch) {
            q.push(' ');
            q.push_str(&self.ret.to_gql());
            q.push_str(&self.modifiers.to_gql());
        }

        q
    }
}

impl Pattern {
    fn to_gql(&self) -> String {
        let mut s = self.node.to_gql();
        if let Some(edge) = &self.edge {
            s.push_str(&edge.to_gql());
            if let Some(node2) = &self.second_node {
                s.push_str(&node2.to_gql());
            } else {
                s.push_str("()");
            }
        }
        s
    }
}

impl NodePattern {
    fn to_gql(&self) -> String {
        let mut s = String::from("(");
        if let Some(v) = &self.variable {
            s.push_str(v.as_str());
        }
        if let Some(l) = &self.label {
            s.push(':');
            s.push_str(l.as_str());
        }
        if !self.props.is_empty() {
            s.push_str(" {");
            let props: Vec<String> = self
                .props
                .iter()
                .take(3) // limit to avoid huge queries
                .map(|(k, v)| format!("{}: {}", k.as_str(), v.to_gql()))
                .collect();
            s.push_str(&props.join(", "));
            s.push('}');
        }
        s.push(')');
        s
    }
}

impl EdgePattern {
    fn to_gql(&self) -> String {
        let inner = match (&self.variable, &self.edge_type) {
            (Some(v), Some(t)) => format!("[{}:{}]", v.as_str(), t.as_str()),
            (None, Some(t)) => format!("[:{}]", t.as_str()),
            (Some(v), None) => format!("[{}]", v.as_str()),
            (None, None) => "[]".to_string(),
        };
        match self.direction {
            Direction::Outgoing => format!("-{inner}->"),
            Direction::Incoming => format!("<-{inner}-"),
            Direction::Undirected => format!("-{inner}-"),
        }
    }
}

impl Filter {
    fn to_gql(&self) -> String {
        format!(
            "{} {} {}",
            self.left.to_gql(),
            self.op.as_str(),
            self.right.to_gql()
        )
    }
}

impl Return {
    fn to_gql(&self) -> String {
        let mut s = String::from("RETURN ");
        if self.distinct {
            s.push_str("DISTINCT ");
        }
        if self.items.is_empty() {
            s.push('*');
        } else {
            let items: Vec<String> = self
                .items
                .iter()
                .take(5)
                .map(|item| item.to_gql())
                .collect();
            s.push_str(&items.join(", "));
        }
        s
    }
}

impl ReturnItem {
    fn to_gql(&self) -> String {
        match self {
            ReturnItem::Star => "*".to_string(),
            ReturnItem::Property(p) => p.to_gql(),
            ReturnItem::Count(None) => "count(*)".to_string(),
            ReturnItem::Count(Some(p)) => format!("count({})", p.to_gql()),
            ReturnItem::Alias(p, name) => format!("{} AS {}", p.to_gql(), name.as_str()),
        }
    }
}

impl Modifiers {
    fn to_gql(&self) -> String {
        let mut s = String::new();
        if let Some(ob) = &self.order_by {
            s.push_str(" ORDER BY ");
            s.push_str(&ob.to_gql());
            if self.order_desc {
                s.push_str(" DESC");
            }
        }
        if let Some(limit) = self.limit {
            if limit > 0 {
                s.push_str(&format!(" LIMIT {limit}"));
            }
        }
        if let Some(skip) = self.skip {
            if skip > 0 {
                s.push_str(&format!(" SKIP {skip}"));
            }
        }
        s
    }
}

impl PropAccess {
    fn to_gql(&self) -> String {
        format!("{}.{}", self.var.as_str(), self.prop.as_str())
    }
}

impl CompOp {
    fn as_str(&self) -> &'static str {
        match self {
            CompOp::Eq => "=",
            CompOp::Neq => "<>",
            CompOp::Lt => "<",
            CompOp::Gt => ">",
            CompOp::Lte => "<=",
            CompOp::Gte => ">=",
        }
    }
}

impl PropValue {
    fn to_gql(&self) -> String {
        match self {
            PropValue::Int(n) => n.to_string(),
            PropValue::Float(f) => {
                if f.is_finite() {
                    format!("{f}")
                } else {
                    "0.0".to_string()
                }
            }
            PropValue::Str(s) => format!("'{}'", s.as_str()),
            PropValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            PropValue::Null => "null".to_string(),
        }
    }
}

impl VarName {
    fn as_str(&self) -> &'static str {
        VARS[(self.0 as usize) % VARS.len()]
    }
}
impl Label {
    fn as_str(&self) -> &'static str {
        LABELS[(self.0 as usize) % LABELS.len()]
    }
}
impl PropName {
    fn as_str(&self) -> &'static str {
        PROPS[(self.0 as usize) % PROPS.len()]
    }
}
impl SmallStr {
    fn as_str(&self) -> &'static str {
        STRINGS[(self.0 as usize) % STRINGS.len()]
    }
}

// ---------------------------------------------------------------------------
// Fuzz target
// ---------------------------------------------------------------------------

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(fq) = FuzzQuery::arbitrary(&mut u) {
        let gql = fq.to_gql();
        // Must never panic
        let _ = grafeo_adapters::query::gql::parse(&gql);
    }
});
