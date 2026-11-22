//! The openCypher TCK's value notation for `values:`, and the observation of a graph for
//! `side_effects:`.
//!
//! A cell is parsed into a [`Cell`] and compared to a result [`Value`] type-exact: an integer
//! never matches a float or a string. Nodes, relationships and paths are recognised by the shape
//! the bindings give them (see [`entity`]).

use std::collections::{BTreeMap, BTreeSet};

use grafeo_common::types::{LogicalType, PropertyKey, Value};

/// A cell in the TCK's value notation.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    /// `null`
    Null,
    /// `true` or `false`
    Bool(bool),
    /// An integer: `42`
    Int(i64),
    /// A float: `1.0`, `1e-3`, `NaN`, `Inf`, `-Inf`
    Float(f64),
    /// A single-quoted string
    String(String),
    /// `[a, b]`
    List(Vec<Cell>),
    /// `{k: v}`
    Map(BTreeMap<String, Cell>),
    /// `(:A:B {p: 1})`
    Node(NodeCell),
    /// `[:T {p: 1}]`
    Relationship(RelationshipCell),
    /// `<(:A)-[:T]->(:B)>`: nodes, relationships, and for each relationship whether it starts
    /// at the node on its left (`->`) or on its right (`<-`)
    Path(Vec<NodeCell>, Vec<RelationshipCell>, Vec<bool>),
}

/// A node cell: its labels and properties.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCell {
    /// The labels, as a set.
    pub labels: BTreeSet<String>,
    /// The properties.
    pub properties: BTreeMap<String, Cell>,
}

/// A relationship cell: its type and properties.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipCell {
    /// The relationship type.
    pub rel_type: String,
    /// The properties.
    pub properties: BTreeMap<String, Cell>,
}

struct Parser<'a> {
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn fail(&self, what: &str) -> String {
        format!("{what} at {} in '{}'", self.at, self.text)
    }

    fn rest(&self) -> &str {
        &self.text[self.at..]
    }

    fn skip(&mut self) {
        let trimmed = self.rest().trim_start();
        self.at = self.text.len() - trimmed.len();
    }

    fn peek(&mut self, token: &str) -> bool {
        self.skip();
        self.rest().starts_with(token)
    }

    fn take(&mut self, token: &str) -> Result<(), String> {
        if !self.peek(token) {
            return Err(self.fail(&format!("expected '{token}'")));
        }
        self.at += token.len();
        Ok(())
    }

    fn name(&mut self) -> Result<String, String> {
        self.skip();
        if self.rest().starts_with('`') {
            let end = self.rest()[1..]
                .find('`')
                .ok_or_else(|| self.fail("unterminated name"))?;
            let name = self.rest()[1..=end].to_string();
            self.at += end + 2;
            return Ok(name);
        }
        let len = self
            .rest()
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(self.rest().len());
        if len == 0 {
            return Err(self.fail("expected a name"));
        }
        let name = self.rest()[..len].to_string();
        self.at += len;
        Ok(name)
    }

    fn word(&mut self, word: &str) -> bool {
        let follows = self
            .rest()
            .strip_prefix(word)
            .map(|after| after.chars().next());
        if follows.is_some_and(|next| !next.is_some_and(char::is_alphanumeric)) {
            self.at += word.len();
            return true;
        }
        false
    }

    fn value(&mut self) -> Result<Cell, String> {
        self.skip();
        if self.word("null") {
            return Ok(Cell::Null);
        }
        if self.word("true") {
            return Ok(Cell::Bool(true));
        }
        if self.word("false") {
            return Ok(Cell::Bool(false));
        }
        for (word, value) in [
            ("NaN", f64::NAN),
            ("-Inf", f64::NEG_INFINITY),
            ("Inf", f64::INFINITY),
        ] {
            if self.rest().starts_with(word) {
                self.at += word.len();
                return Ok(Cell::Float(value));
            }
        }
        match self.rest().chars().next() {
            Some('\'') => self.string().map(Cell::String),
            Some('{') => self.map().map(Cell::Map),
            Some('(') => self.node().map(Cell::Node),
            Some('<') => self.path(),
            Some('[') if self.rest().starts_with("[:") => {
                self.relationship().map(Cell::Relationship)
            }
            Some('[') => self.list().map(Cell::List),
            _ => self.number(),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.take("'")?;
        let mut out = String::new();
        let mut chars = self.rest().char_indices();
        while let Some((offset, c)) = chars.next() {
            match c {
                '\'' => {
                    self.at += offset + 1;
                    return Ok(out);
                }
                '\\' => match chars.next() {
                    Some((_, 'n')) => out.push('\n'),
                    Some((_, 't')) => out.push('\t'),
                    Some((_, 'r')) => out.push('\r'),
                    Some((_, 'b')) => out.push('\u{8}'),
                    Some((_, 'f')) => out.push('\u{c}'),
                    Some((start, 'u')) => {
                        let hex = self.rest().get(start + 1..start + 5).unwrap_or("");
                        let code = u32::from_str_radix(hex, 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or_else(|| self.fail("bad \\u escape"))?;
                        out.push(code);
                        for _ in 0..4 {
                            chars.next();
                        }
                    }
                    Some((_, other @ ('\'' | '"' | '\\'))) => out.push(other),
                    Some((_, other)) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => break,
                },
                other => out.push(other),
            }
        }
        Err(self.fail("unterminated string"))
    }

    fn number(&mut self) -> Result<Cell, String> {
        let len = self
            .rest()
            .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
            .unwrap_or(self.rest().len());
        let text = &self.rest()[..len];
        if text.is_empty() {
            return Err(self.fail("expected a value"));
        }
        let cell = if text.contains(['.', 'e', 'E']) {
            Cell::Float(text.parse().map_err(|_| self.fail("bad float"))?)
        } else {
            Cell::Int(text.parse().map_err(|_| self.fail("bad integer"))?)
        };
        self.at += len;
        Ok(cell)
    }

    fn items<T>(
        &mut self,
        close: &str,
        mut item: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let mut out = Vec::new();
        if self.peek(close) {
            self.take(close)?;
            return Ok(out);
        }
        loop {
            out.push(item(self)?);
            if self.peek(",") {
                self.take(",")?;
                continue;
            }
            self.take(close)?;
            return Ok(out);
        }
    }

    fn list(&mut self) -> Result<Vec<Cell>, String> {
        self.take("[")?;
        self.items("]", Self::value)
    }

    fn map(&mut self) -> Result<BTreeMap<String, Cell>, String> {
        self.take("{")?;
        let entries = self.items("}", |p| {
            let key = p.name()?;
            p.take(":")?;
            Ok((key, p.value()?))
        })?;
        Ok(entries.into_iter().collect())
    }

    fn properties(&mut self) -> Result<BTreeMap<String, Cell>, String> {
        if self.peek("{") {
            self.map()
        } else {
            Ok(BTreeMap::new())
        }
    }

    fn node(&mut self) -> Result<NodeCell, String> {
        self.take("(")?;
        let mut labels = BTreeSet::new();
        while self.peek(":") {
            self.take(":")?;
            labels.insert(self.name()?);
        }
        let properties = self.properties()?;
        self.take(")")?;
        Ok(NodeCell { labels, properties })
    }

    fn relationship(&mut self) -> Result<RelationshipCell, String> {
        self.take("[")?;
        self.take(":")?;
        let rel_type = self.name()?;
        let properties = self.properties()?;
        self.take("]")?;
        Ok(RelationshipCell {
            rel_type,
            properties,
        })
    }

    fn path(&mut self) -> Result<Cell, String> {
        self.take("<")?;
        let mut nodes = vec![self.node()?];
        let (mut relationships, mut forward) = (Vec::new(), Vec::new());
        while !self.peek(">") {
            if self.peek("<-") {
                self.take("<-")?;
                relationships.push(self.relationship()?);
                self.take("-")?;
                forward.push(false);
            } else {
                self.take("-")?;
                relationships.push(self.relationship()?);
                self.take("->")?;
                forward.push(true);
            }
            nodes.push(self.node()?);
        }
        self.take(">")?;
        Ok(Cell::Path(nodes, relationships, forward))
    }
}

/// Parses one cell of the TCK's value notation.
///
/// # Errors
///
/// What is wrong with the cell and where.
pub fn parse(text: &str) -> Result<Cell, String> {
    let mut parser = Parser { text, at: 0 };
    let cell = parser.value()?;
    parser.skip();
    if parser.at != text.len() {
        return Err(parser.fail("unexpected text"));
    }
    Ok(cell)
}

/// A node or relationship as a result gives it.
#[derive(Debug)]
pub enum Entity<'a> {
    /// A node: its id, labels and properties.
    Node(i64, BTreeSet<String>, Vec<(&'a str, &'a Value)>),
    /// A relationship: its id, type, source, target and properties.
    Relationship(i64, &'a str, i64, i64, Vec<(&'a str, &'a Value)>),
}

/// A cell type with the record wrapper taken off: `Records(shape)` is a column of `shape`.
fn resolved(ty: &LogicalType) -> &LogicalType {
    match ty {
        LogicalType::Records(shape) => resolved(shape),
        other => other,
    }
}

/// The type of a map's value at `key`: by key in a struct, the one type of a map.
fn member<'t>(ty: &'t LogicalType, key: &str) -> &'t LogicalType {
    match resolved(ty) {
        LogicalType::Struct(fields) => fields
            .iter()
            .find(|(name, _)| name == key)
            .map_or(&LogicalType::Any, |(_, ty)| ty),
        LogicalType::Map { value, .. } => value,
        _ => &LogicalType::Any,
    }
}

/// The type of a list's item at `index`: the one type of a list, by position in a tuple.
fn item(ty: &LogicalType, index: usize) -> &LogicalType {
    match resolved(ty) {
        LogicalType::List(item) => item,
        LogicalType::Tuple(items) => items.get(index).unwrap_or(&LogicalType::Any),
        _ => &LogicalType::Any,
    }
}

/// The entity `value` is where its cell type `ty` says so (`QueryResult::cell_type`): a map at
/// type `Node` is a node, one at type `Edge` a relationship, and anywhere else a map is a map,
/// whatever its keys. The underscore keys of an entity are not properties.
#[must_use]
pub fn entity<'a>(value: &'a Value, ty: &LogicalType) -> Option<Entity<'a>> {
    let Value::Map(map) = value else {
        return None;
    };
    let kind = resolved(ty);
    if !matches!(kind, LogicalType::Node | LogicalType::Edge) {
        return None;
    }
    let get = |key: &str| map.get(&PropertyKey::new(key));
    let Some(Value::Int64(id)) = get("_id") else {
        return None;
    };
    let properties = |reserved: &[&str]| {
        map.iter()
            .filter(|(key, _)| !reserved.contains(&key.as_str()))
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>()
    };
    if let (LogicalType::Node, Some(Value::List(labels))) = (kind, get("_labels")) {
        let labels = labels
            .iter()
            .filter_map(|label| match label {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        return Some(Entity::Node(*id, labels, properties(&["_id", "_labels"])));
    }
    if let (
        LogicalType::Edge,
        Some(Value::String(rel_type)),
        Some(Value::Int64(source)),
        Some(Value::Int64(target)),
    ) = (kind, get("_type"), get("_source"), get("_target"))
    {
        return Some(Entity::Relationship(
            *id,
            rel_type.as_str(),
            *source,
            *target,
            properties(&["_id", "_type", "_source", "_target"]),
        ));
    }
    None
}

/// How cells compare: whether every list is a bag. (The TCK's dialect, a string cell matching a
/// temporal, is the caller's: it writes temporals as their text first, see
/// `temporals_as_text`.)
#[derive(Debug, Clone, Copy, Default)]
pub struct Flags {
    /// Compare every list as a bag.
    pub unordered_lists: bool,
}

/// Whether every item of `expected` pairs with its own item of `actual` under `pairs`, by an
/// exact bipartite matching (a greedy one fails `[(), (:A)]` against `[(:A), (:B)]`).
pub fn bag_matches<E, A>(expected: &[E], actual: &[A], pairs: impl Fn(&E, &A) -> bool) -> bool {
    fn assign(
        e: usize,
        edges: &[Vec<usize>],
        seen: &mut [bool],
        owner: &mut [Option<usize>],
    ) -> bool {
        for &a in &edges[e] {
            if !seen[a] {
                seen[a] = true;
                if owner[a].is_none_or(|other| assign(other, edges, seen, owner)) {
                    owner[a] = Some(e);
                    return true;
                }
            }
        }
        false
    }
    if expected.len() != actual.len() {
        return false;
    }
    let edges: Vec<Vec<usize>> = expected
        .iter()
        .map(|e| {
            (0..actual.len())
                .filter(|&a| pairs(e, &actual[a]))
                .collect()
        })
        .collect();
    let mut owner = vec![None; actual.len()];
    (0..expected.len()).all(|e| assign(e, &edges, &mut vec![false; actual.len()], &mut owner))
}

/// Whether `actual`, a map's or an entity's entries, holds the properties `expected` writes;
/// `ty` is the type of the map, whose values take the types inside it.
fn properties_match(
    expected: &BTreeMap<String, Cell>,
    actual: &[(&str, &Value)],
    ty: &LogicalType,
    flags: Flags,
) -> bool {
    expected.len() == actual.len()
        && actual.iter().all(|(key, value)| {
            expected
                .get(*key)
                .is_some_and(|cell| matches(cell, value, member(ty, key), flags))
        })
}

fn node_matches(expected: &NodeCell, actual: &Value, ty: &LogicalType, flags: Flags) -> bool {
    matches!(entity(actual, ty), Some(Entity::Node(_, labels, properties))
        if labels == expected.labels
            && properties_match(&expected.properties, &properties, &LogicalType::Any, flags))
}

fn relationship_matches(
    expected: &RelationshipCell,
    actual: &Value,
    ty: &LogicalType,
    flags: Flags,
) -> bool {
    matches!(entity(actual, ty), Some(Entity::Relationship(_, rel_type, _, _, properties))
        if rel_type == expected.rel_type
            && properties_match(&expected.properties, &properties, &LogicalType::Any, flags))
}

fn path_matches(
    (nodes, relationships, forward): (&[NodeCell], &[RelationshipCell], &[bool]),
    actual: &Value,
    flags: Flags,
) -> bool {
    let Value::Path {
        nodes: actual_nodes,
        edges,
    } = actual
    else {
        return false;
    };
    if actual_nodes.len() != nodes.len() || edges.len() != relationships.len() {
        return false;
    }
    // A path's nodes are nodes and its edges relationships, by the type of a path
    let id = |value: &Value| match entity(value, &LogicalType::Node) {
        Some(Entity::Node(id, ..)) => Some(id),
        _ => None,
    };
    let ends_hold = edges.iter().enumerate().all(|(i, edge)| {
        let (start, end) = if forward[i] {
            (&actual_nodes[i], &actual_nodes[i + 1])
        } else {
            (&actual_nodes[i + 1], &actual_nodes[i])
        };
        matches!(entity(edge, &LogicalType::Edge), Some(Entity::Relationship(_, _, source, target, _))
            if Some(source) == id(start) && Some(target) == id(end))
    });
    ends_hold
        && nodes
            .iter()
            .zip(actual_nodes.iter())
            .all(|(cell, node)| node_matches(cell, node, &LogicalType::Node, flags))
        && relationships
            .iter()
            .zip(edges.iter())
            .all(|(cell, edge)| relationship_matches(cell, edge, &LogicalType::Edge, flags))
}

/// Whether `actual`, of cell type `ty`, is the value `expected` writes, type-exact.
#[must_use]
pub fn matches(expected: &Cell, actual: &Value, ty: &LogicalType, flags: Flags) -> bool {
    match (expected, actual) {
        (Cell::Null, Value::Null) => true,
        (Cell::Bool(e), Value::Bool(a)) => e == a,
        (Cell::Int(e), Value::Int64(a)) => e == a,
        (Cell::Float(e), Value::Float64(a)) => e == a || (e.is_nan() && a.is_nan()),
        (Cell::String(e), Value::String(a)) => e.as_str() == a.as_str(),
        (Cell::List(e), Value::List(a)) => {
            let typed: Vec<(&Value, &LogicalType)> = a
                .iter()
                .enumerate()
                .map(|(i, v)| (v, item(ty, i)))
                .collect();
            let pairs =
                |cell: &Cell, (value, ty): &(&Value, &LogicalType)| matches(cell, value, ty, flags);
            if flags.unordered_lists {
                bag_matches(e, &typed, pairs)
            } else {
                e.len() == typed.len() && e.iter().zip(typed.iter()).all(|(c, v)| pairs(c, v))
            }
        }
        (Cell::Map(e), Value::Map(map)) => {
            entity(actual, ty).is_none()
                && properties_match(
                    e,
                    &map.iter().map(|(k, v)| (k.as_str(), v)).collect::<Vec<_>>(),
                    ty,
                    flags,
                )
        }
        (Cell::Node(e), _) => node_matches(e, actual, ty, flags),
        (Cell::Relationship(e), _) => relationship_matches(e, actual, ty, flags),
        (Cell::Path(nodes, relationships, forward), _) => {
            path_matches((nodes, relationships, forward), actual, flags)
        }
        _ => false,
    }
}

/// What a graph holds, as the TCK observes side effects: node and relationship ids, the label
/// names in use, and the properties as (entity, key, value) triples.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Observation {
    /// The node ids.
    pub nodes: BTreeSet<i64>,
    /// The relationship ids.
    pub relationships: BTreeSet<i64>,
    /// The label names on any node.
    pub labels: BTreeSet<String>,
    /// The properties: whether on a node, the entity's id, the key, and the typed value.
    pub properties: BTreeSet<(bool, i64, String, String)>,
}

impl Observation {
    /// Adds what one row's entity holds; `ty` is its cell type.
    pub fn add(&mut self, value: &Value, ty: &LogicalType) {
        match entity(value, ty) {
            Some(Entity::Node(id, labels, properties)) => {
                self.nodes.insert(id);
                self.labels.extend(labels);
                for (key, value) in properties {
                    self.properties
                        .insert((true, id, key.to_string(), format!("{value:?}")));
                }
            }
            Some(Entity::Relationship(id, .., properties)) => {
                self.relationships.insert(id);
                for (key, value) in properties {
                    self.properties
                        .insert((false, id, key.to_string(), format!("{value:?}")));
                }
            }
            None => {}
        }
    }

    /// The side effects from `self` to `after`, each TCK key and its count.
    #[must_use]
    pub fn side_effects(&self, after: &Self) -> BTreeMap<&'static str, usize> {
        fn changes<T: Ord>(before: &BTreeSet<T>, after: &BTreeSet<T>) -> (usize, usize) {
            (
                after.difference(before).count(),
                before.difference(after).count(),
            )
        }
        let mut out = BTreeMap::new();
        for (plus, minus, (added, removed)) in [
            ("+nodes", "-nodes", changes(&self.nodes, &after.nodes)),
            (
                "+relationships",
                "-relationships",
                changes(&self.relationships, &after.relationships),
            ),
            ("+labels", "-labels", changes(&self.labels, &after.labels)),
            (
                "+properties",
                "-properties",
                changes(&self.properties, &after.properties),
            ),
        ] {
            out.insert(plus, added);
            out.insert(minus, removed);
        }
        out
    }
}

/// Splits a `values:` table into its rows of cell texts; `\|` is a pipe inside a cell.
#[must_use]
pub fn table_rows(table: &str) -> Vec<Vec<String>> {
    table
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let inner = line
                .strip_prefix('|')
                .and_then(|l| l.strip_suffix('|'))
                .unwrap_or(line);
            let (mut cells, mut cell) = (Vec::new(), String::new());
            let mut chars = inner.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '\\' if chars.peek() == Some(&'|') => {
                        cell.push('|');
                        chars.next();
                    }
                    '|' => cells.push(std::mem::take(&mut cell).trim().to_string()),
                    other => cell.push(other),
                }
            }
            cells.push(cell.trim().to_string());
            cells
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_engine::GrafeoDB;
    use serde_json::Value as Json;

    use super::*;

    fn map(entries: impl IntoIterator<Item = (String, Value)>) -> Value {
        Value::Map(Arc::new(
            entries
                .into_iter()
                .map(|(k, v)| (PropertyKey::new(k), v))
                .collect(),
        ))
    }

    fn properties(json: &Json, db: &GrafeoDB) -> Vec<(String, Value)> {
        json.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), neutral(v, db)))
            .collect()
    }

    fn node(json: &Json, db: &GrafeoDB) -> Value {
        let labels = json["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| Value::String(l.as_str().unwrap().into()))
            .collect::<Vec<_>>();
        let mut entries = vec![
            (
                "_id".to_string(),
                Value::Int64(json["id"].as_i64().unwrap()),
            ),
            ("_labels".to_string(), Value::List(labels.into())),
        ];
        entries.extend(properties(&json["properties"], db));
        map(entries)
    }

    fn relationship(json: &Json, db: &GrafeoDB) -> Value {
        let mut entries = vec![
            (
                "_id".to_string(),
                Value::Int64(json["id"].as_i64().unwrap()),
            ),
            (
                "_type".to_string(),
                Value::String(json["type"].as_str().unwrap().into()),
            ),
            (
                "_source".to_string(),
                Value::Int64(json["source"].as_i64().unwrap()),
            ),
            (
                "_target".to_string(),
                Value::Int64(json["target"].as_i64().unwrap()),
            ),
        ];
        entries.extend(properties(&json["properties"], db));
        map(entries)
    }

    /// A result value from the neutral form of `values_cases.json`, in the shape results give
    /// it (a node and a relationship as maps, a `cypher` value from the engine itself).
    fn neutral(json: &Json, db: &GrafeoDB) -> Value {
        match json {
            Json::Null => Value::Null,
            Json::Bool(b) => Value::Bool(*b),
            Json::Array(items) => Value::List(items.iter().map(|i| neutral(i, db)).collect()),
            Json::Object(object) => {
                let (kind, inner) = object.iter().next().unwrap();
                match kind.as_str() {
                    "int" => Value::Int64(inner.as_str().unwrap().parse().unwrap()),
                    "float" => Value::Float64(match inner.as_str().unwrap() {
                        "NaN" => f64::NAN,
                        "Inf" => f64::INFINITY,
                        "-Inf" => f64::NEG_INFINITY,
                        text => text.parse().unwrap(),
                    }),
                    "string" => Value::String(inner.as_str().unwrap().into()),
                    "map" => map(properties(inner, db)),
                    "node" => node(inner, db),
                    "relationship" => relationship(inner, db),
                    "path" => Value::Path {
                        nodes: inner["nodes"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|n| node(n, db))
                            .collect(),
                        edges: inner["relationships"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|r| relationship(r, db))
                            .collect(),
                    },
                    "cypher" => {
                        let query = format!("RETURN {} AS v", inner.as_str().unwrap());
                        let result = db.execute_cypher(&query).expect(&query);
                        result.rows()[0][0].clone()
                    }
                    other => panic!("unknown neutral kind {other}"),
                }
            }
            other => panic!("not a neutral value: {other}"),
        }
    }

    /// The cell type the engine would give a neutral value: a node, a relationship and a path
    /// by their kind, the items of a list and a map by position and key, a `cypher` value the
    /// engine's own.
    fn neutral_type(json: &Json, db: &GrafeoDB) -> LogicalType {
        match json {
            Json::Array(items) => {
                LogicalType::Tuple(items.iter().map(|i| neutral_type(i, db)).collect())
            }
            Json::Object(object) => {
                let (kind, inner) = object.iter().next().unwrap();
                match kind.as_str() {
                    "map" => LogicalType::Struct(
                        inner
                            .as_object()
                            .unwrap()
                            .iter()
                            .map(|(k, v)| (k.clone(), neutral_type(v, db)))
                            .collect(),
                    ),
                    "node" => LogicalType::Node,
                    "relationship" => LogicalType::Edge,
                    "path" => LogicalType::Path,
                    "cypher" => {
                        let query = format!("RETURN {} AS v", inner.as_str().unwrap());
                        db.execute_cypher(&query)
                            .expect(&query)
                            .cell_type(0, 0)
                            .clone()
                    }
                    _ => LogicalType::Any,
                }
            }
            _ => LogicalType::Any,
        }
    }

    #[test]
    fn the_shared_value_cases() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/spec/runners/values_cases.json"
        );
        let file: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let db = GrafeoDB::new_in_memory();
        let mut wrong = Vec::new();
        for case in file["cases"].as_array().unwrap() {
            let cell = parse(case["cell"].as_str().unwrap()).unwrap();
            let flags = Flags {
                unordered_lists: case["unordered_lists"].as_bool().unwrap_or(false),
            };
            let mut actual = neutral(&case["actual"], &db);
            if case["dialect"].as_str() == Some("tck") {
                actual = crate::temporals_as_text(&db, &actual);
            }
            let verdict = matches(&cell, &actual, &neutral_type(&case["actual"], &db), flags);
            // A gap of this runner gives the opposite verdict until fixed
            let gap = case["runner_gaps"].get("rust").is_some();
            let want = case["match"].as_bool().unwrap() != gap;
            if verdict != want {
                wrong.push(format!("{} vs {}: {verdict}", case["cell"], case["actual"]));
            }
        }
        assert!(
            wrong.is_empty(),
            "cases that disagree:\n{}",
            wrong.join("\n")
        );
    }

    #[test]
    fn the_shared_round_trips() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/spec/runners/values_cases.json"
        );
        let file: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let db = GrafeoDB::new_in_memory();
        let mut wrong = Vec::new();
        for trip in file["round_trips"].as_array().unwrap() {
            let value = neutral(&serde_json::json!({ "cypher": trip["cypher"] }), &db);
            let text = crate::temporals_as_text(&db, &value);
            let holds = text == Value::String(trip["text"].as_str().unwrap().into());
            // A strict expected failure where the runner's gap names this runner
            if holds == trip["runner_gaps"].get("rust").is_some() {
                wrong.push(format!("{}: {text:?}", trip["cypher"]));
            }
        }
        assert!(
            wrong.is_empty(),
            "round trips that disagree:\n{}",
            wrong.join("\n")
        );
    }

    #[test]
    fn malformed_cells_are_refused() {
        for text in [
            "'open",
            "[1, 2",
            "1 2",
            "(:A",
            "{k 1}",
            "",
            "<(:A)-[:T]-(:B)>",
        ] {
            assert!(parse(text).is_err(), "{text} parsed");
        }
    }

    #[test]
    fn side_effects_count_set_differences() {
        let mut before = Observation::default();
        before.nodes.extend([1, 2]);
        before.labels.insert("A".into());
        before
            .properties
            .insert((true, 1, "p".into(), "Int64(1)".into()));
        let mut after = Observation::default();
        after.nodes.extend([2, 3]);
        after.labels.extend(["A".to_string(), "B".to_string()]);
        after
            .properties
            .insert((true, 1, "p".into(), "Int64(2)".into()));
        let effects = before.side_effects(&after);
        assert_eq!(effects["+nodes"], 1);
        assert_eq!(effects["-nodes"], 1);
        assert_eq!(effects["+labels"], 1);
        assert_eq!(effects["-labels"], 0);
        assert_eq!((effects["+properties"], effects["-properties"]), (1, 1));
    }

    #[test]
    fn tables_split_on_unescaped_pipes() {
        let rows = table_rows("| n | m |\n| 'a \\| b' | 1 |\n");
        assert_eq!(rows, vec![vec!["n", "m"], vec!["'a | b'", "1"]]);
    }

    const FILE: &str = r#"meta:
  language: cypher
  values_dialect: tck

tests:
  - name: one
    setup_gql:
      - "CREATE PROCEDURE `t.p`() RETURNS () AS { UNWIND [] AS r FINISH }"
    query: RETURN 1 AS n
    expect:
      unordered_lists: true
      values: |
        | n |
        | 1 |
      side_effects:
        +nodes: 2
        -labels: 1
"#;

    #[test]
    fn the_tck_keys_parse() {
        let file = crate::gtest::parse_gtest_text("x.gtest", FILE).unwrap();
        assert_eq!(file.meta.values_dialect, "tck");
        let case = &file.tests[0];
        assert_eq!(case.setup_gql.len(), 1);
        assert_eq!(case.expect.values.as_deref(), Some("| n |\n| 1 |"));
        assert!(case.expect.unordered_lists);
        assert_eq!(
            case.expect.side_effects,
            Some(vec![("+nodes".to_string(), 2), ("-labels".to_string(), 1)])
        );
    }

    #[test]
    fn no_side_effects_and_the_default_dialect() {
        let text = FILE.replace("  values_dialect: tck\n", "").replace(
            "      side_effects:\n        +nodes: 2\n        -labels: 1\n",
            "      side_effects: {}\n",
        );
        let file = crate::gtest::parse_gtest_text("x.gtest", &text).unwrap();
        assert_eq!(file.meta.values_dialect, "strict");
        assert_eq!(file.tests[0].expect.side_effects, Some(Vec::new()));
    }

    #[test]
    fn a_runner_gap_parses_and_names_known_runners() {
        let gap = r#"    runner_gap:
      runners: [python, dart]
      issue: fix in 0.6.4
      reason: "temporals"
    query:"#;
        let text = FILE.replace("    query:", gap);
        let file = crate::gtest::parse_gtest_text("x.gtest", &text).unwrap();
        let parsed = file.tests[0].runner_gap.clone().unwrap();
        assert_eq!(parsed.runners, ["python", "dart"]);
        assert_eq!(
            (parsed.issue.as_str(), parsed.reason.as_str()),
            ("fix in 0.6.4", "temporals")
        );
        for (from, to) in [
            ("python, dart", "python, cobol"),
            ("      issue: fix in 0.6.4\n", ""),
        ] {
            let wrong = text.replace(from, to);
            assert!(
                crate::gtest::parse_gtest_text("x.gtest", &wrong).is_err(),
                "{to}"
            );
        }
    }

    #[test]
    fn wrong_tck_keys_are_refused() {
        for (from, to) in [
            ("+nodes: 2", "+edges: 2"),
            ("+nodes: 2", "+nodes: two"),
            ("values_dialect: tck", "values_dialect: loose"),
            ("unordered_lists: true", "empty: true"),
        ] {
            let text = FILE.replace(from, to);
            assert!(
                crate::gtest::parse_gtest_text("x.gtest", &text).is_err(),
                "{to} was accepted"
            );
        }
    }
}
