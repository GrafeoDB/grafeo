//! Upserts: create or update nodes and edges by a key property.
//!
//! Each call writes with one `UNWIND $rows ... MERGE ... SET ...` statement
//! (an edge upsert looks up its endpoints first), so the rows are checked,
//! logged, reported to CDC and counted like any query. Rows
//! apply in order and see the writes of the rows before them: a key repeated
//! within one call behaves like repeated calls (the first creates, the next
//! update).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::graph_handle::GraphHandle;
use super::{GrafeoDB, QueryResult};
use crate::session::Session;

/// At most this many skipped row indices are reported.
const MAX_SKIPPED_ROWS: usize = 1_000;

/// What an upsert did with its rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpsertSummary {
    /// Rows that created a node or edge.
    pub created: usize,
    /// Rows that updated an existing node or edge.
    pub updated: usize,
    /// Rows that were not written: a row without its key, an edge row
    /// without a source or target field, and an edge row whose endpoint key
    /// matches no node or more than one node.
    pub skipped: usize,
    /// The indices of the skipped rows, in order (at most 1,000).
    pub skipped_rows: Vec<usize>,
}

/// How [`GrafeoDB::upsert_edges`] finds edges and their endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeUpsertOptions {
    /// The property that identifies an edge between two nodes (default `id`).
    pub key: String,
    /// The node property the endpoint fields hold (default `id`). A property
    /// index on it makes the lookups fast.
    pub endpoint_key: String,
    /// Labels an endpoint must have (default none: the key alone).
    pub endpoint_labels: Vec<String>,
    /// The row field with the source node's key (default `src`).
    pub src_field: String,
    /// The row field with the target node's key (default `dst`).
    pub dst_field: String,
    /// Whether an edge's properties become exactly the row's (default
    /// `false`: the row's properties are merged into the edge's).
    pub replace: bool,
}

impl Default for EdgeUpsertOptions {
    fn default() -> Self {
        Self {
            key: "id".to_string(),
            endpoint_key: "id".to_string(),
            endpoint_labels: Vec::new(),
            src_field: "src".to_string(),
            dst_field: "dst".to_string(),
            replace: false,
        }
    }
}

/// Creates or updates one node per row, matched by `key` and all of `labels`.
///
/// With `replace`, a node's properties become exactly the row's (key
/// included); without, the row's properties are merged into the node's and
/// none is removed. Labels are never removed.
fn upsert_nodes(
    run: impl FnOnce(&str, HashMap<String, Value>) -> Result<QueryResult>,
    labels: &[&str],
    key: &str,
    rows: Vec<HashMap<PropertyKey, Value>>,
    replace: bool,
) -> Result<UpsertSummary> {
    check_name("key", key)?;
    let label_pattern = labels
        .iter()
        .map(|label| check_name("label", label).map(|()| format!(":{}", quote(label))))
        .collect::<Result<String>>()?;
    let total = rows.len();
    let key_property = PropertyKey::new(key);
    let items = rows
        .into_iter()
        .enumerate()
        .filter(|(_, row)| row.get(&key_property).is_some_and(|value| !value.is_null()))
        .map(|(index, row)| item(index, [("row", map_value(row))]))
        .collect();
    let set = if replace { "=" } else { "+=" };
    let query = format!(
        "UNWIND $rows AS item \
         MERGE (n{label_pattern} {{{key}: item.row.{key}}}) \
         SET n {set} item.row \
         RETURN item.i",
        key = quote(key),
    );
    summarize(run, &query, items, total, |counters| counters.nodes_created)
}

/// Creates or updates one edge per row between the nodes whose
/// `options.endpoint_key` is the row's source and target field, matched by
/// its type and `options.key`. Every other field of a row is an edge
/// property. A row is skipped when it lacks the key, the source field or the
/// target field, or when no node or more than one node has its endpoint
/// key; endpoints are never created.
fn upsert_edges(
    session: &Session,
    edge_type: &str,
    rows: Vec<HashMap<PropertyKey, Value>>,
    options: &EdgeUpsertOptions,
) -> Result<UpsertSummary> {
    check_name("edge type", edge_type)?;
    for (what, name) in [
        ("key", &options.key),
        ("endpoint key", &options.endpoint_key),
        ("source field", &options.src_field),
        ("target field", &options.dst_field),
    ] {
        check_name(what, name)?;
    }
    let (key_field, src_field, dst_field) = (&options.key, &options.src_field, &options.dst_field);
    if key_field == src_field || key_field == dst_field || src_field == dst_field {
        return Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!(
                "upsert: the key ({key_field}), source field ({src_field}) and target field \
                 ({dst_field}) must be different fields"
            ),
        )));
    }
    let endpoint_labels = options
        .endpoint_labels
        .iter()
        .map(|label| check_name("label", label).map(|()| format!(":{}", quote(label))))
        .collect::<Result<String>>()?;
    let total = rows.len();
    let (src, dst) = (
        PropertyKey::new(options.src_field.as_str()),
        PropertyKey::new(options.dst_field.as_str()),
    );
    let key = PropertyKey::new(options.key.as_str());
    let mut items: Vec<(usize, Value)> = rows
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut row)| {
            let source = row.remove(&src).filter(|value| !value.is_null())?;
            let target = row.remove(&dst).filter(|value| !value.is_null())?;
            if row.get(&key).is_none_or(Value::is_null) {
                return None;
            }
            Some((
                index,
                item(
                    index,
                    [("src", source), ("dst", target), ("props", map_value(row))],
                ),
            ))
        })
        .collect();
    let set = if options.replace { "=" } else { "+=" };
    let query = format!(
        "UNWIND $rows AS item \
         MATCH (s{endpoint_labels} {{{endpoint_key}: item.src}}), \
               (d{endpoint_labels} {{{endpoint_key}: item.dst}}) \
         MERGE (s)-[r:{edge_type} {{{key}: item.props.{key}}}]->(d) \
         SET r {set} item.props \
         RETURN item.i, id(s), id(d)",
        endpoint_key = quote(&options.endpoint_key),
        edge_type = quote(edge_type),
        key = quote(&options.key),
    );

    // A row whose endpoint key more than one node has matches one pair of
    // endpoints per node and comes back once per pair (a row can also come
    // back once per edge of an existing pair, which is not ambiguous). Such
    // an attempt is undone and the call runs again without those rows, so
    // they write nothing; each attempt drops at least one row.
    let result = loop {
        if items.is_empty() {
            return Ok(summary(total, &BTreeSet::new(), 0));
        }
        let mut ambiguous = BTreeSet::new();
        let attempt = session.as_one_write(|| {
            let rows = items
                .iter()
                .map(|(_, item)| item.clone())
                .collect::<Vec<_>>();
            let result = session.execute_with_params(
                &query,
                HashMap::from([("rows".to_string(), Value::List(rows.into()))]),
            )?;
            ambiguous = rows_with_several_endpoint_pairs(&result);
            if ambiguous.is_empty() {
                Ok(result)
            } else {
                Err(Error::Internal("upsert: ambiguous endpoint keys".into()))
            }
        });
        match attempt {
            Ok(result) => break result,
            Err(_) if !ambiguous.is_empty() => {
                items.retain(|(index, _)| !ambiguous.contains(index));
            }
            Err(error) => return Err(error),
        }
    };
    Ok(count(&result, total, result.counters.edges_created))
}

/// The row indices the upsert statement returned, from its first column.
fn returned_rows(result: &QueryResult) -> impl Iterator<Item = usize> + '_ {
    result.rows().iter().filter_map(|row| match row.first() {
        Some(Value::Int64(index)) => usize::try_from(*index).ok(),
        _ => None,
    })
}

/// The row indices the edge upsert statement returned with more than one
/// pair of endpoints (its second and third columns).
fn rows_with_several_endpoint_pairs(result: &QueryResult) -> BTreeSet<usize> {
    let mut pairs: HashMap<usize, BTreeSet<(i64, i64)>> = HashMap::new();
    for row in result.rows() {
        if let (Some(Value::Int64(index)), Some(Value::Int64(source)), Some(Value::Int64(target))) =
            (row.first(), row.get(1), row.get(2))
            && let Ok(index) = usize::try_from(*index)
        {
            pairs.entry(index).or_default().insert((*source, *target));
        }
    }
    pairs
        .into_iter()
        .filter(|(_, endpoints)| endpoints.len() > 1)
        .map(|(index, _)| index)
        .collect()
}

/// Runs the upsert statement over `items` and counts what it did with the
/// `total` rows (see [`count`]).
fn summarize(
    run: impl FnOnce(&str, HashMap<String, Value>) -> Result<QueryResult>,
    query: &str,
    items: Vec<Value>,
    total: usize,
    created: impl Fn(&super::WriteCounters) -> u64,
) -> Result<UpsertSummary> {
    if items.is_empty() {
        return Ok(summary(total, &BTreeSet::new(), 0));
    }
    let result = run(
        query,
        HashMap::from([("rows".to_string(), Value::List(items.into()))]),
    )?;
    Ok(count(&result, total, created(&result.counters)))
}

/// What the upsert statement did with the `total` rows: the rows it returns
/// were written, `created` of them new.
fn count(result: &QueryResult, total: usize, created: u64) -> UpsertSummary {
    let written: BTreeSet<usize> = returned_rows(result).collect();
    summary(
        total,
        &written,
        usize::try_from(created).unwrap_or(usize::MAX),
    )
}

fn summary(total: usize, written: &BTreeSet<usize>, created: usize) -> UpsertSummary {
    let created = created.min(written.len());
    let skipped_rows: Vec<usize> = (0..total)
        .filter(|index| !written.contains(index))
        .collect();
    UpsertSummary {
        created,
        updated: written.len() - created,
        skipped: skipped_rows.len(),
        skipped_rows: skipped_rows.into_iter().take(MAX_SKIPPED_ROWS).collect(),
    }
}

/// One row of the statement's `$rows`: its index and its fields.
fn item<const N: usize>(index: usize, fields: [(&str, Value); N]) -> Value {
    let mut map = BTreeMap::new();
    map.insert(
        PropertyKey::new("i"),
        Value::Int64(i64::try_from(index).unwrap_or(i64::MAX)),
    );
    for (name, value) in fields {
        map.insert(PropertyKey::new(name), value);
    }
    Value::Map(map.into())
}

fn map_value(row: HashMap<PropertyKey, Value>) -> Value {
    Value::Map(row.into_iter().collect::<BTreeMap<_, _>>().into())
}

/// A name quoted as a GQL identifier.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn check_name(what: &str, name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!("upsert: the {what} must not be empty"),
        )));
    }
    Ok(())
}

impl GrafeoDB {
    /// Creates or updates one node per row in the current graph, matched by
    /// `key` and all of `labels`, in one statement.
    ///
    /// With `replace`, a node's properties become exactly the row's (key
    /// included); without, the row's properties are merged into the node's
    /// and none is removed. Labels are never removed. Rows apply in order: a
    /// key repeated within one call creates one node, which the later rows
    /// update. A row without the key is skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if a row breaks a constraint or the schema; nothing
    /// of the call is written then.
    pub fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> Result<UpsertSummary> {
        upsert_nodes(
            |query, params| self.execute_with_params(query, params),
            labels,
            key,
            rows,
            replace,
        )
    }

    /// Creates or updates one edge of `edge_type` per row in the current
    /// graph, between the nodes the row's source and target fields name, in
    /// one statement (see [`EdgeUpsertOptions`]).
    ///
    /// A row is skipped when it lacks the edge key, the source field or the
    /// target field, or when no node or more than one node has its endpoint
    /// key; endpoints are never created. Rows
    /// apply in order: a key repeated within one call creates one edge, which
    /// the later rows update.
    ///
    /// # Errors
    ///
    /// Returns an error if a row breaks the schema, nothing of the call is
    /// written then, or if the edge key, source field and target field are
    /// not three different fields.
    pub fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> Result<UpsertSummary> {
        upsert_edges(&self.session(), edge_type, rows, options)
    }
}

impl GraphHandle<'_> {
    /// [`GrafeoDB::upsert_nodes`] in this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or a row breaks a
    /// constraint or the schema.
    pub fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> Result<UpsertSummary> {
        upsert_nodes(
            |query, params| self.execute_with_params(query, params),
            labels,
            key,
            rows,
            replace,
        )
    }

    /// [`GrafeoDB::upsert_edges`] in this graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph no longer exists or a row breaks the
    /// schema.
    pub fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> Result<UpsertSummary> {
        upsert_edges(&self.session()?, edge_type, rows, options)
    }
}

impl Session {
    /// [`GrafeoDB::upsert_nodes`] in the session's graph, inside its
    /// transaction when one is open.
    ///
    /// # Errors
    ///
    /// Returns an error if a row breaks a constraint or the schema.
    pub fn upsert_nodes(
        &self,
        labels: &[&str],
        key: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        replace: bool,
    ) -> Result<UpsertSummary> {
        upsert_nodes(
            |query, params| self.execute_with_params(query, params),
            labels,
            key,
            rows,
            replace,
        )
    }

    /// [`GrafeoDB::upsert_edges`] in the session's graph, inside its
    /// transaction when one is open.
    ///
    /// # Errors
    ///
    /// Returns an error if a row breaks the schema.
    pub fn upsert_edges(
        &self,
        edge_type: &str,
        rows: Vec<HashMap<PropertyKey, Value>>,
        options: &EdgeUpsertOptions,
    ) -> Result<UpsertSummary> {
        upsert_edges(self, edge_type, rows, options)
    }
}
