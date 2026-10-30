//! The native loader: typed CSV files as nodes and edges, in bulk.
//!
//! [`NodeLoad`] reads one node per record with the given labels, each column
//! a property of its [`ColumnKind`]; [`EdgeLoad`] reads one edge per record,
//! its ends found by the value of a key column among the nodes of a label.
//! The file is parsed here (the quoting rules of `LOAD CSV`: quoted fields
//! with the delimiter, escaped quotes and line breaks inside, a BOM, `\r\n`),
//! each field converted straight to its column's value, and the rows go to
//! the store's bulk write (see [`GraphWriter::create_nodes`]): one write of
//! the whole file, in one transaction, all or nothing.
//!
//! The same load runs from Rust and Python (`GrafeoDB::load_nodes_csv`,
//! `load_edges_csv`, and on sessions in their transaction) and from any
//! query language (`CALL grafeo.load_nodes_csv({...})`).
//!
//! # Stability: Experimental
//!
//! New in 0.6.1; formats (Parquet, JSON Lines), sources (bytes) and options
//! may be added.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use arcstr::ArcStr;
use grafeo_common::types::{
    Date, LogicalType, NodeId, PropertyKey, Time, Timestamp, Value, ZonedDatetime,
};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    GraphWriter, HashKey, NewEdge, Operator, OperatorError, OperatorResult,
};
use grafeo_core::execution::vector::ValueVector;

/// How a column's text becomes a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ColumnKind {
    /// The text as it is.
    #[default]
    String,
    /// A 64-bit integer (`42`, `-7`).
    Int64,
    /// A 64-bit float (`9.5`, `7`, `1e3`).
    Float64,
    /// `true` or `false`, in any case.
    Bool,
    /// A date, `YYYY-MM-DD`.
    Date,
    /// A timestamp, `YYYY-MM-DDTHH:MM:SS[.fff]`, with an offset (`Z`,
    /// `+01:00`, read as the instant) or without one.
    Timestamp,
    /// Not stored: a column the file has and the graph does not want.
    Skip,
}

impl ColumnKind {
    /// The kind a name gives, in any case: `string`, `int64`, `float64`,
    /// `bool`, `date`, `timestamp`, `skip` (and `integer`, `int`, `float`,
    /// `double`, `boolean`, `text`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "string" | "text" => Self::String,
            "int64" | "integer" | "int" => Self::Int64,
            "float64" | "float" | "double" => Self::Float64,
            "bool" | "boolean" => Self::Bool,
            "date" => Self::Date,
            "timestamp" => Self::Timestamp,
            "skip" => Self::Skip,
            _ => return None,
        })
    }

    /// The kind's name, as errors give it.
    fn name(self) -> &'static str {
        match self {
            Self::String => "STRING",
            Self::Int64 => "INT64",
            Self::Float64 => "FLOAT64",
            Self::Bool => "BOOL",
            Self::Date => "DATE",
            Self::Timestamp => "TIMESTAMP",
            Self::Skip => "SKIP",
        }
    }
}

/// A column of a file and the kind of its values.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Column {
    /// The column's name: its header, or the name of the property it fills
    /// in a file without a header (where the columns follow the file's
    /// order).
    pub name: String,
    /// How its text becomes a value.
    pub kind: ColumnKind,
}

/// A file of nodes: one node per record, with `labels`, each column a
/// property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeLoad {
    /// The labels of every node.
    pub labels: Vec<String>,
    /// The key column: present in every record and unique in the file (a
    /// `UNIQUE` constraint on it also keeps it unique among the nodes the
    /// graph has).
    pub key: Option<String>,
    /// The kinds of the columns: by name with a header (a column not listed
    /// is a string), in file order without one. Empty: every column a
    /// string.
    pub columns: Vec<Column>,
    /// The field separator.
    pub delimiter: u8,
    /// Whether the first record names the columns.
    pub header: bool,
    /// Fields that stand for no value, besides an empty one (`NULL`, `\N`).
    pub nulls: Vec<String>,
}

impl Default for NodeLoad {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            key: None,
            columns: Vec::new(),
            delimiter: b',',
            header: true,
            nulls: Vec::new(),
        }
    }
}

/// How an end of the edges of an [`EdgeLoad`] is found: the node of
/// `label` whose `property` equals the record's value in `column` (compared
/// as values of their kind: the text `1` does not find the integer 1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EndKey {
    /// The label of the nodes it is one of.
    pub label: String,
    /// The property that holds their keys.
    pub property: String,
    /// The column of the file that holds the record's key.
    pub column: String,
}

/// What a record whose end no node has becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MissingEnd {
    /// An error naming the key and its line; nothing of the file is loaded.
    #[default]
    Error,
    /// No edge; the record counts as skipped.
    Skip,
}

/// A file of edges: one edge of `edge_type` per record, from the node
/// `source` finds to the one `target` finds, each other column a property.
/// The key columns are not stored on the edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeLoad {
    /// The type of every edge.
    pub edge_type: String,
    /// How the source of each edge is found.
    pub source: EndKey,
    /// How the target of each edge is found.
    pub target: EndKey,
    /// The kinds of the columns, as in [`NodeLoad::columns`] (the key
    /// columns' kinds too: a key read as text finds only text keys).
    pub columns: Vec<Column>,
    /// The field separator.
    pub delimiter: u8,
    /// Whether the first record names the columns.
    pub header: bool,
    /// Fields that stand for no value, besides an empty one.
    pub nulls: Vec<String>,
    /// What a record whose end no node has becomes.
    pub missing: MissingEnd,
}

impl Default for EdgeLoad {
    fn default() -> Self {
        Self {
            edge_type: String::new(),
            source: EndKey::default(),
            target: EndKey::default(),
            columns: Vec::new(),
            delimiter: b',',
            header: true,
            nulls: Vec::new(),
            missing: MissingEnd::default(),
        }
    }
}

/// What a load did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadReport {
    /// The records the file has (the header not counted).
    pub rows: u64,
    /// The nodes or edges created.
    pub created: u64,
    /// The records that created nothing ([`MissingEnd::Skip`]).
    pub skipped: u64,
}

/// Loads the nodes of the file at `path` as `spec` describes, through
/// `writer` (one bulk write, in the writer's transaction).
///
/// # Errors
///
/// A file it cannot read, a value its column's kind cannot read, a missing
/// or repeated key, and the errors of the write (a constraint).
pub(crate) fn load_nodes(
    writer: &GraphWriter,
    path: &Path,
    spec: &NodeLoad,
) -> Result<LoadReport, OperatorError> {
    let text = read_file(path)?;
    let mut records = Records::new(&text, spec.delimiter);
    let layout = Layout::of(&mut records, spec.header, &spec.columns, path)?;
    let key = spec
        .key
        .as_deref()
        .map(|key| layout.position(key, path))
        .transpose()?;
    let mut seen_keys: HashMap<(std::mem::Discriminant<Value>, HashKey), u64> = HashMap::new();
    let mut rows = Vec::new();
    let mut count = 0_u64;
    while let Some((line, fields)) = records.next_record()? {
        count += 1;
        let values = layout.values(&fields, line, &spec.nulls)?;
        if let Some(position) = key {
            let name = &layout.columns[position].0;
            let Some(value) = values[position].as_ref() else {
                return Err(data_error(format!(
                    "the key column '{name}' has no value at line {line}"
                )));
            };
            if let Some(first) = seen_keys.insert(exact_key(value), line) {
                return Err(data_error(format!(
                    "the key {value} of column '{name}' is at line {first} and at line {line}"
                )));
            }
        }
        rows.push(layout.properties(values, &[]));
    }
    let created = writer.create_nodes(&spec.labels, rows)?;
    Ok(LoadReport {
        rows: count,
        created: created.len() as u64,
        skipped: 0,
    })
}

/// Loads the edges of the file at `path` as `spec` describes, through
/// `writer` (one bulk write, in the writer's transaction). The ends are
/// found in a map of each end's keys, built once from its label's nodes at
/// the writer's view.
///
/// # Errors
///
/// As [`load_nodes`], and a key no node has with [`MissingEnd::Error`].
pub(crate) fn load_edges(
    writer: &GraphWriter,
    path: &Path,
    spec: &EdgeLoad,
) -> Result<LoadReport, OperatorError> {
    let text = read_file(path)?;
    let mut records = Records::new(&text, spec.delimiter);
    let layout = Layout::of(&mut records, spec.header, &spec.columns, path)?;
    let source_column = layout.position(&spec.source.column, path)?;
    let target_column = layout.position(&spec.target.column, path)?;
    let source_keys = key_map(writer, &spec.source)?;
    let same_keys =
        spec.source.label == spec.target.label && spec.source.property == spec.target.property;
    let target_keys = if same_keys {
        None
    } else {
        Some(key_map(writer, &spec.target)?)
    };
    let target_keys = target_keys.as_ref().unwrap_or(&source_keys);
    let mut edges = Vec::new();
    let (mut count, mut skipped) = (0_u64, 0_u64);
    while let Some((line, fields)) = records.next_record()? {
        count += 1;
        let values = layout.values(&fields, line, &spec.nulls)?;
        let ends = [
            (source_column, &source_keys, &spec.source),
            (target_column, target_keys, &spec.target),
        ]
        .map(|(position, keys, end)| {
            values[position]
                .as_ref()
                .and_then(|value| keys.get(&exact_key(value)).copied())
                .ok_or((position, end))
        });
        match ends {
            [Ok(src), Ok(dst)] => edges.push(NewEdge {
                src,
                dst,
                edge_type: spec.edge_type.clone(),
                properties: layout.properties(values, &[source_column, target_column]),
            }),
            [Err((position, end)), _] | [_, Err((position, end))] => {
                if spec.missing == MissingEnd::Skip {
                    skipped += 1;
                    continue;
                }
                let key = values[position].as_ref().map_or_else(
                    || "no value".to_string(),
                    |value| format!("the key {value}"),
                );
                return Err(data_error(format!(
                    "line {line}: no node with label '{}' has {key} in '{}' (column '{}')",
                    end.label, end.property, end.column
                )));
            }
        }
    }
    let created = writer.create_edges(edges)?;
    Ok(LoadReport {
        rows: count,
        created: created.len() as u64,
        skipped,
    })
}

/// The nodes of `end`'s label that the writer sees, by their value of its
/// property (compared exactly, see [`exact_key`]).
fn key_map(
    writer: &GraphWriter,
    end: &EndKey,
) -> Result<HashMap<(std::mem::Discriminant<Value>, HashKey), NodeId>, OperatorError> {
    let store = writer.store();
    let view = writer.view();
    let nodes = store.nodes_by_label(&end.label, view)?;
    let mut keys = HashMap::with_capacity(nodes.len());
    for node in nodes {
        if let Some(value) = store.node_property(node, &end.property, view)? {
            keys.entry(exact_key(&value)).or_insert(node);
        }
    }
    Ok(keys)
}

/// A key that tells values of different kinds apart (a date and an
/// integer of the same number, a float and an integer of the same bits):
/// [`HashKey`] alone joins some of them.
fn exact_key(value: &Value) -> (std::mem::Discriminant<Value>, HashKey) {
    (std::mem::discriminant(value), HashKey::from_value(value))
}

/// The text of the file at `path`, without a BOM.
fn read_file(path: &Path) -> Result<String, OperatorError> {
    let bytes = std::fs::read(path).map_err(|error| {
        OperatorError::Execution(format!("cannot read '{}': {error}", path.display()))
    })?;
    let mut text = String::from_utf8(bytes).map_err(|error| {
        OperatorError::Execution(format!("'{}' is not UTF-8 text: {error}", path.display()))
    })?;
    if text.starts_with('\u{feff}') {
        text.drain(..'\u{feff}'.len_utf8());
    }
    Ok(text)
}

/// An error of the file's data (a value, a key): GQLSTATUS 22018, as a cast
/// that cannot read its text.
fn data_error(message: String) -> OperatorError {
    OperatorError::from(Error::from(QueryError::new(
        QueryErrorKind::InvalidCharacterValueForCast,
        message,
    )))
}

/// The columns of a file: each record field's name and kind, in file order.
struct Layout {
    columns: Vec<(String, ColumnKind)>,
}

impl Layout {
    /// The layout from the header (read from `records`) or, without one,
    /// from `columns` in file order.
    fn of(
        records: &mut Records<'_>,
        header: bool,
        columns: &[Column],
        path: &Path,
    ) -> Result<Self, OperatorError> {
        if !header {
            if columns.is_empty() {
                return Err(OperatorError::Execution(format!(
                    "'{}' has no header: the load needs its columns",
                    path.display()
                )));
            }
            return Ok(Self {
                columns: columns
                    .iter()
                    .map(|column| (column.name.clone(), column.kind))
                    .collect(),
            });
        }
        let names = records
            .next_record()?
            .map(|(_, fields)| fields.into_iter().map(Cow::into_owned).collect::<Vec<_>>())
            .unwrap_or_default();
        for column in columns {
            if !names.contains(&column.name) {
                return Err(OperatorError::Execution(format!(
                    "'{}' has no column '{}'",
                    path.display(),
                    column.name
                )));
            }
        }
        let columns = names
            .into_iter()
            .map(|name| {
                let kind = columns
                    .iter()
                    .find(|column| column.name == name)
                    .map_or(ColumnKind::String, |column| column.kind);
                (name, kind)
            })
            .collect();
        Ok(Self { columns })
    }

    /// The position of the column `name`.
    fn position(&self, name: &str, path: &Path) -> Result<usize, OperatorError> {
        self.columns
            .iter()
            .position(|(column, _)| column == name)
            .ok_or_else(|| {
                OperatorError::Execution(format!("'{}' has no column '{name}'", path.display()))
            })
    }

    /// The values of a record's fields, by column (none for an empty field,
    /// a null marker and a skipped column).
    fn values(
        &self,
        fields: &[Cow<'_, str>],
        line: u64,
        nulls: &[String],
    ) -> Result<Vec<Option<Value>>, OperatorError> {
        self.columns
            .iter()
            .enumerate()
            .map(|(position, (name, kind))| {
                let Some(field) = fields.get(position) else {
                    return Ok(None);
                };
                if *kind == ColumnKind::Skip
                    || field.is_empty()
                    || nulls.iter().any(|null| null == field.as_ref())
                {
                    return Ok(None);
                }
                convert(field, *kind).map(Some).ok_or_else(|| {
                    data_error(format!(
                        "line {line}: '{field}' in column '{name}' is no {}",
                        kind.name()
                    ))
                })
            })
            .collect()
    }

    /// The properties of a record: its values with their column names, but
    /// those at `leave_out` (an edge's key columns).
    fn properties(&self, values: Vec<Option<Value>>, leave_out: &[usize]) -> Vec<(String, Value)> {
        values
            .into_iter()
            .enumerate()
            .filter(|(position, _)| !leave_out.contains(position))
            .filter_map(|(position, value)| Some((self.columns[position].0.clone(), value?)))
            .collect()
    }
}

/// The value `kind` reads from `field`, if it can.
fn convert(field: &str, kind: ColumnKind) -> Option<Value> {
    Some(match kind {
        ColumnKind::String | ColumnKind::Skip => Value::String(ArcStr::from(field)),
        ColumnKind::Int64 => Value::Int64(field.trim().parse().ok()?),
        ColumnKind::Float64 => Value::Float64(field.trim().parse().ok()?),
        ColumnKind::Bool => {
            let field = field.trim();
            if field.eq_ignore_ascii_case("true") {
                Value::Bool(true)
            } else if field.eq_ignore_ascii_case("false") {
                Value::Bool(false)
            } else {
                return None;
            }
        }
        ColumnKind::Date => Value::Date(Date::parse(field.trim())?),
        ColumnKind::Timestamp => Value::Timestamp(timestamp(field.trim())?),
    })
}

/// A timestamp from `YYYY-MM-DDTHH:MM:SS[.fff]`, with an offset (the
/// instant it names) or without one (or a space for the `T`).
fn timestamp(text: &str) -> Option<Timestamp> {
    if let Some(zoned) = ZonedDatetime::parse(text) {
        return Some(zoned.as_timestamp());
    }
    let split = text.find(['T', 't', ' '])?;
    let date = Date::parse(&text[..split])?;
    let time = Time::parse(&text[split + 1..])?;
    Some(Timestamp::from_date_time(date, time))
}

/// The records of a CSV text, each with the line it starts on.
struct Records<'a> {
    text: &'a str,
    at: usize,
    line: u64,
    delimiter: u8,
}

impl<'a> Records<'a> {
    fn new(text: &'a str, delimiter: u8) -> Self {
        Self {
            text,
            at: 0,
            line: 0,
            delimiter,
        }
    }

    /// The next record and the line it starts on (counting from 1), or
    /// none at the end. A field borrows the text unless it has an escaped
    /// quote.
    fn next_record(&mut self) -> Result<Option<(u64, Vec<Cow<'a, str>>)>, OperatorError> {
        let bytes = self.text.as_bytes();
        // An empty line between records is no record.
        while self.at < bytes.len() && matches!(bytes[self.at], b'\n' | b'\r') {
            if bytes[self.at] == b'\n' {
                self.line += 1;
            }
            self.at += 1;
        }
        if self.at >= bytes.len() {
            return Ok(None);
        }
        self.line += 1;
        let start_line = self.line;
        let mut fields = Vec::new();
        loop {
            // One field, from `self.at`.
            if bytes.get(self.at) == Some(&b'"') {
                let open = self.at + 1;
                let mut i = open;
                let mut escaped = false;
                loop {
                    match bytes.get(i) {
                        None => {
                            return Err(OperatorError::Execution(format!(
                                "an unterminated quoted field starts at line {start_line}"
                            )));
                        }
                        Some(b'"') if bytes.get(i + 1) == Some(&b'"') => {
                            escaped = true;
                            i += 2;
                        }
                        Some(b'"') => break,
                        Some(b'\n') => {
                            self.line += 1;
                            i += 1;
                        }
                        Some(_) => i += 1,
                    }
                }
                let raw = &self.text[open..i];
                fields.push(if escaped {
                    Cow::Owned(raw.replace("\"\"", "\""))
                } else {
                    Cow::Borrowed(raw)
                });
                self.at = i + 1;
            } else {
                let start = self.at;
                let mut i = start;
                while i < bytes.len()
                    && bytes[i] != self.delimiter
                    && !matches!(bytes[i], b'\n' | b'\r')
                {
                    i += 1;
                }
                fields.push(Cow::Borrowed(&self.text[start..i]));
                self.at = i;
            }
            match bytes.get(self.at) {
                Some(&byte) if byte == self.delimiter => self.at += 1,
                Some(b'\r') => {
                    self.at += 1;
                    if bytes.get(self.at) == Some(&b'\n') {
                        self.at += 1;
                    }
                    return Ok(Some((start_line, fields)));
                }
                Some(b'\n') => {
                    self.at += 1;
                    return Ok(Some((start_line, fields)));
                }
                None => return Ok(Some((start_line, fields))),
                Some(_) => {
                    return Err(OperatorError::Execution(format!(
                        "line {}: text after a quoted field",
                        self.line
                    )));
                }
            }
        }
    }
}

// === The procedures: CALL grafeo.load_nodes_csv({...}), load_edges_csv ===

/// The procedures that load: `grafeo.load_nodes_csv` and
/// `grafeo.load_edges_csv`.
pub(crate) const NODES_PROCEDURE: &str = "grafeo.load_nodes_csv";
/// See [`NODES_PROCEDURE`].
pub(crate) const EDGES_PROCEDURE: &str = "grafeo.load_edges_csv";

/// The columns a load procedure yields.
pub(crate) const REPORT_COLUMNS: [&str; 3] = ["rows", "created", "skipped"];

/// What a load procedure loads, from its one map argument.
pub(crate) enum LoadCall {
    /// `grafeo.load_nodes_csv`.
    Nodes(PathBuf, NodeLoad),
    /// `grafeo.load_edges_csv`.
    Edges(PathBuf, EdgeLoad),
}

impl LoadCall {
    /// The load `procedure` (one of the two names) describes with
    /// `options`: `path`, then for nodes `labels` (a string or a list),
    /// `key`; for edges `type`, `source` and `target` (maps of `label`,
    /// `property`, `column`), `missing` (`'error'` or `'skip'`); and for both
    /// `types` (column to type name, in any case), `columns` (the column
    /// names in file order, for a file without a header), `delimiter`,
    /// `header`, `nulls`.
    ///
    /// # Errors
    ///
    /// A missing or unknown option, or one of another type.
    pub(crate) fn from_options(
        procedure: &str,
        options: &BTreeMap<PropertyKey, Value>,
    ) -> Result<Self, Error> {
        let nodes = procedure == NODES_PROCEDURE;
        let allowed: &[&str] = if nodes {
            &[
                "path",
                "labels",
                "key",
                "types",
                "columns",
                "delimiter",
                "header",
                "nulls",
            ]
        } else {
            &[
                "path",
                "type",
                "source",
                "target",
                "missing",
                "types",
                "columns",
                "delimiter",
                "header",
                "nulls",
            ]
        };
        if let Some(unknown) = options.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(option_error(
                procedure,
                unknown.as_str(),
                &format!("is no option; the options are {}", allowed.join(", ")),
            ));
        }
        let option = |name: &str| options.get(&PropertyKey::new(name));
        let path = PathBuf::from(
            text(procedure, "path", option("path"))?
                .ok_or_else(|| option_error(procedure, "path", "is required"))?,
        );
        let header = match option("header") {
            None => true,
            Some(Value::Bool(header)) => *header,
            Some(_) => return Err(option_error(procedure, "header", "must be a boolean")),
        };
        let delimiter = match text(procedure, "delimiter", option("delimiter"))? {
            None => b',',
            Some(text) if text.len() == 1 => text.as_bytes()[0],
            Some(_) => {
                return Err(option_error(
                    procedure,
                    "delimiter",
                    "must be one character",
                ));
            }
        };
        let nulls = texts(procedure, "nulls", option("nulls"))?.unwrap_or_default();
        let columns = columns(procedure, option("types"), option("columns"), header)?;
        if nodes {
            let labels = texts(procedure, "labels", option("labels"))?
                .filter(|labels| !labels.is_empty())
                .ok_or_else(|| option_error(procedure, "labels", "is required"))?;
            let key = text(procedure, "key", option("key"))?;
            return Ok(Self::Nodes(
                path,
                NodeLoad {
                    labels,
                    key,
                    columns,
                    delimiter,
                    header,
                    nulls,
                },
            ));
        }
        let edge_type = text(procedure, "type", option("type"))?
            .ok_or_else(|| option_error(procedure, "type", "is required"))?;
        let missing = match text(procedure, "missing", option("missing"))?.as_deref() {
            None | Some("error") => MissingEnd::Error,
            Some("skip") => MissingEnd::Skip,
            Some(_) => return Err(option_error(procedure, "missing", "is 'error' or 'skip'")),
        };
        Ok(Self::Edges(
            path,
            EdgeLoad {
                edge_type,
                source: end_key(procedure, "source", option("source"))?,
                target: end_key(procedure, "target", option("target"))?,
                columns,
                delimiter,
                header,
                nulls,
                missing,
            },
        ))
    }

    /// Runs the load through `writer`.
    fn run(&self, writer: &GraphWriter) -> Result<LoadReport, OperatorError> {
        match self {
            Self::Nodes(path, spec) => load_nodes(writer, path, spec),
            Self::Edges(path, spec) => load_edges(writer, path, spec),
        }
    }
}

/// The error of option `option` of `procedure`.
fn option_error(procedure: &str, option: &str, problem: &str) -> Error {
    Error::from(QueryError::new(
        QueryErrorKind::InvalidValueType,
        format!("Option '{option}' of {procedure} {problem}"),
    ))
}

/// A text option, if given.
fn text(procedure: &str, option: &str, value: Option<&Value>) -> Result<Option<String>, Error> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.to_string())),
        Some(_) => Err(option_error(procedure, option, "must be a string")),
    }
}

/// A list-of-texts option (one text counts as a list of one), if given.
fn texts(
    procedure: &str,
    option: &str,
    value: Option<&Value>,
) -> Result<Option<Vec<String>>, Error> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(vec![text.to_string()])),
        Some(Value::List(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => Ok(text.to_string()),
                _ => Err(option_error(procedure, option, "must be a list of strings")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(option_error(procedure, option, "must be a list of strings")),
    }
}

/// The columns of `types` (name to type name) and `columns` (the names in
/// file order, which a file without a header needs).
fn columns(
    procedure: &str,
    types: Option<&Value>,
    order: Option<&Value>,
    header: bool,
) -> Result<Vec<Column>, Error> {
    let kinds: Vec<(String, ColumnKind)> = match types {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Map(types)) => types
            .iter()
            .map(|(name, kind)| {
                let kind = match kind {
                    Value::String(kind) => ColumnKind::from_name(kind),
                    _ => None,
                }
                .ok_or_else(|| {
                    option_error(
                        procedure,
                        "types",
                        &format!(
                            "gives column '{name}' no type it knows (string, int64, float64,                              bool, date, timestamp, skip)"
                        ),
                    )
                })?;
                Ok((name.as_str().to_string(), kind))
            })
            .collect::<Result<_, Error>>()?,
        Some(_) => return Err(option_error(procedure, "types", "must be a map")),
    };
    let kind_of = |name: &str| {
        kinds
            .iter()
            .find(|(column, _)| column == name)
            .map_or(ColumnKind::String, |(_, kind)| *kind)
    };
    match texts(procedure, "columns", order)? {
        Some(names) => Ok(names
            .into_iter()
            .map(|name| Column {
                kind: kind_of(&name),
                name,
            })
            .collect()),
        None if !header && !kinds.is_empty() => Err(option_error(
            procedure,
            "columns",
            "is required for a file without a header: the names in file order",
        )),
        None => Ok(kinds
            .into_iter()
            .map(|(name, kind)| Column { name, kind })
            .collect()),
    }
}

/// An end option: a map of `label`, `property` and `column`.
fn end_key(procedure: &str, option: &str, value: Option<&Value>) -> Result<EndKey, Error> {
    let Some(Value::Map(end)) = value else {
        return Err(option_error(
            procedure,
            option,
            "is required: a map of label, property and column",
        ));
    };
    let part = |name: &str| {
        text(procedure, option, end.get(&PropertyKey::new(name)))?
            .ok_or_else(|| option_error(procedure, option, &format!("needs '{name}'")))
    };
    Ok(EndKey {
        label: part("label")?,
        property: part("property")?,
        column: part("column")?,
    })
}

/// The operator of a load procedure: on its first pull it runs the load
/// through the statement's writer and gives one row of the report, in the
/// columns the call yields (indices into [`REPORT_COLUMNS`]).
pub(crate) struct LoadCallOperator {
    load: LoadCall,
    writer: GraphWriter,
    yielded: Vec<usize>,
    done: bool,
}

impl LoadCallOperator {
    pub(crate) fn new(load: LoadCall, writer: GraphWriter, yielded: Vec<usize>) -> Self {
        Self {
            load,
            writer,
            yielded,
            done: false,
        }
    }
}

impl Operator for LoadCallOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let report = self.load.run(&self.writer)?;
        let numbers = [report.rows, report.created, report.skipped];
        let columns = self
            .yielded
            .iter()
            .map(|&index| {
                let mut column = ValueVector::with_capacity(LogicalType::Int64, 1);
                column.push_int64(i64::try_from(numbers[index]).unwrap_or(i64::MAX));
                column
            })
            .collect();
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "LoadCsv"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
