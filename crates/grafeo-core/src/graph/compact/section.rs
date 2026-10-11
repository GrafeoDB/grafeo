//! The `CompactStore` section of a 0.5.x database file, read for migration.
//!
//! From 0.5.40 to 0.5.44, `compact()` wrote the default graph's compacted
//! base as one raw chunk holding encoding version 1 or 3: the magic `GCST`, the
//! version, a flags byte (bit 0: the base keeps the ids of its nodes and
//! edges), the node tables (one per label set, each with its property
//! columns and zone maps), the relationship tables (forward and backward CSR
//! adjacency and edge property columns), the id maps when the base keeps the
//! ids, and a CRC32 of all of it. Nothing writes this section any more: the
//! fold reads it once, as a 0.5.x file migrates (see [`fold`](super::fold)).
//! Removed in 0.7.0 with the other 0.5.x readers.

use bytes::Bytes;
use grafeo_common::storage::section::SectionType;
use grafeo_common::storage::{SectionSource, legacy_bytes};
use grafeo_common::types::{EdgeId, NodeId, PropertyKey};
use grafeo_common::utils::error::Error;
use grafeo_common::utils::hash::FxHashMap;

use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{EdgeSchema, TableSchema};
use super::{CompactStore, label_set_key, labels_of_unescaped_key};

/// Magic bytes identifying a CompactStore encoding.
const MAGIC: [u8; 4] = *b"GCST";

/// v3 layout: columns in blocks, each with a zone map. Written by 0.5.42 to
/// 0.5.44.
const FORMAT_VERSION_V3: u8 = 3;

/// v1 layout: columns whole. Written by 0.5.40 and 0.5.41 (`compact()` of
/// 0.5.32 to 0.5.39 kept the base in memory only).
const FORMAT_VERSION_V1: u8 = 1;

/// The encodings a 0.5.x file holds. Version 2 (blocks without zone maps)
/// was never released.
const READABLE_ENCODINGS: [u8; 2] = [FORMAT_VERSION_V1, FORMAT_VERSION_V3];

/// Reads the compacted base of a 0.5.x file from its `CompactStore` section:
/// one raw chunk holding encoding version 1 or 3.
///
/// # Errors
///
/// Returns [`Error::Corruption`] naming the section when its bytes do not
/// decode, and [`Error::Serialization`] when the section is not one raw
/// chunk: only a 0.6.0 development build wrote it otherwise.
pub(super) fn read_base(
    source: &dyn SectionSource,
) -> grafeo_common::utils::error::Result<CompactStore> {
    let Some(bytes) = legacy_bytes(source).map_err(|e| in_section(SectionType::CompactStore, e))?
    else {
        return Err(Error::Serialization(
            "section CompactStore: not the 0.5.x encoding (a 0.6.0 development build wrote \
             it); recreate the database with this build"
                .to_string(),
        ));
    };
    deserialize_compact_store(&bytes, &READABLE_ENCODINGS)
        .map_err(|e| Error::corruption(format!("section CompactStore: {e}")))
}

/// `error` with `section_type` in front of its message when it is an
/// [`Error::Corruption`] or an [`Error::Serialization`]; any other error as
/// it is.
pub(super) fn in_section(section_type: SectionType, error: Error) -> Error {
    match error {
        Error::Corruption(_) | Error::Serialization(_) => {
            error.wrapped(format_args!("section {section_type:?}"))
        }
        other => other,
    }
}

/// What an id map needs of its ids: node ids and edge ids.
trait MapId: Copy + Eq + std::hash::Hash {
    /// The id that marks a reverse slot without an entry; never a real id.
    const INVALID: Self;
    /// What the ids name, for messages.
    const WHAT: &'static str;

    fn from_raw(raw: u64) -> Self;

    fn raw(self) -> u64;
}

impl MapId for NodeId {
    const INVALID: Self = NodeId::INVALID;
    const WHAT: &'static str = "node";

    fn from_raw(raw: u64) -> Self {
        NodeId::new(raw)
    }

    fn raw(self) -> u64 {
        self.as_u64()
    }
}

impl MapId for EdgeId {
    const INVALID: Self = EdgeId::INVALID;
    const WHAT: &'static str = "edge";

    fn from_raw(raw: u64) -> Self {
        EdgeId::new(raw)
    }

    fn raw(self) -> u64 {
        self.as_u64()
    }
}

// ── Deserialization ────────────────────────────────────────────────

/// Reads a column body in the layout of encoding `version`: v1 whole
/// ([`ColumnCodec::read_from`]), v3 in blocks ([`ColumnCodec::read_blocked`]).
fn read_column(data: &Bytes, pos: &mut usize, version: u8) -> Result<ColumnCodec, String> {
    match version {
        FORMAT_VERSION_V1 => ColumnCodec::read_from(data, pos),
        FORMAT_VERSION_V3 => ColumnCodec::read_blocked(data, pos),
        _ => return Err(format!("unsupported CompactStore version {version}")),
    }
    .map_err(str::to_string)
}

/// Reads a store from `data_bytes`, an encoding of one of the versions
/// `accepted`.
fn deserialize_compact_store(
    data_bytes: &bytes::Bytes,
    accepted: &[u8],
) -> Result<CompactStore, String> {
    let data: &[u8] = data_bytes.as_ref();
    if data.len() < 10 {
        return Err("data too short for CompactStore section".into());
    }

    // Verify CRC32.
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(format!(
            "CRC32 mismatch: stored {stored_crc:#010X}, computed {computed_crc:#010X}"
        ));
    }

    let mut pos = 0;

    // Header.
    if data[pos..pos + 4] != MAGIC {
        return Err("bad magic".into());
    }
    pos += 4;
    let version = data[pos];
    pos += 1;
    if !accepted.contains(&version) {
        return Err(format!(
            "unsupported CompactStore section version {version} (supported here: {accepted:?})"
        ));
    }
    let flags = data[pos];
    pos += 1;
    let preserves_ids = flags & 0x01 != 0;

    // Node tables. Every count read below is untrusted: it is checked
    // against the bytes left before anything is allocated for it.
    let num_node_tables = read_u32(data, &mut pos)? as usize;
    check_table_count(num_node_tables, "node tables")?;
    fits(
        num_node_tables,
        NODE_TABLE_MIN_BYTES,
        data,
        pos,
        "node tables",
    )?;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    let mut table_labels: Vec<Vec<arcstr::ArcStr>> = Vec::with_capacity(num_node_tables);
    // Each table is named by the labels of its nodes, joined with `|`
    // without escapes (see `labels_of_unescaped_key`).

    for table_idx in 0..num_node_tables {
        let table_id = u16::try_from(table_idx).map_err(|_| "node table id overflow")?;
        let labels = labels_of_unescaped_key(&read_string(data, &mut pos)?);
        let row_count = read_u32(data, &mut pos)? as usize;
        let num_cols = read_u32(data, &mut pos)? as usize;
        fits(num_cols, COLUMN_MIN_BYTES, data, pos, "columns")?;

        let mut columns: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();

        for _ in 0..num_cols {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);

            // The column's zone map, which the fold does not use.
            let has_zm = *data.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            if has_zm == 1 {
                skip_zone_map(data, &mut pos)?;
            }

            let column =
                read_column(data_bytes, &mut pos, version).map_err(|e| format!("codec: {e}"))?;
            columns.insert(key, column);
        }

        let schema = TableSchema::new(label_set_key(&labels), table_id);
        let table = NodeTable::from_columns(schema, columns, row_count);
        node_tables.push(table);
        table_labels.push(labels);
    }
    let node_rows: Vec<usize> = node_tables.iter().map(NodeTable::len).collect();

    // Relationship tables.
    let num_rel_tables = read_u32(data, &mut pos)? as usize;
    check_table_count(num_rel_tables, "relationship tables")?;
    fits(
        num_rel_tables,
        REL_TABLE_MIN_BYTES,
        data,
        pos,
        "relationship tables",
    )?;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    let mut rel_table_id_to_type: Vec<arcstr::ArcStr> = Vec::with_capacity(num_rel_tables);

    for rel_idx in 0..num_rel_tables {
        let rel_table_id = u16::try_from(rel_idx).map_err(|_| "relationship table id overflow")?;
        let edge_type = read_string(data, &mut pos)?;
        let edge_type = arcstr::ArcStr::from(edge_type.as_str());
        let src_tid = read_u16(data, &mut pos)?;
        let dst_tid = read_u16(data, &mut pos)?;
        let (Some(&src_rows), Some(&dst_rows)) = (
            node_rows.get(usize::from(src_tid)),
            node_rows.get(usize::from(dst_tid)),
        ) else {
            return Err(format!(
                "relationship table {rel_idx} ({edge_type}) joins tables {src_tid} and \
                 {dst_tid} of {}",
                node_rows.len()
            ));
        };

        let fwd = CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("fwd CSR: {e}"))?;
        check_adjacency(&fwd, src_rows, dst_rows, rel_idx, "forward")?;

        let has_bwd = *data.get(pos).ok_or("truncated bwd flag")?;
        pos += 1;
        let bwd = if has_bwd == 1 {
            let bwd =
                CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("bwd CSR: {e}"))?;
            check_adjacency(&bwd, dst_rows, src_rows, rel_idx, "backward")?;
            if bwd.num_edges() != fwd.num_edges() {
                return Err(format!(
                    "relationship table {rel_idx} has {} edges backward and {} forward",
                    bwd.num_edges(),
                    fwd.num_edges()
                ));
            }
            if let Some(&position) = bwd
                .edge_data()
                .and_then(|positions| positions.iter().find(|&&p| p as usize >= fwd.num_edges()))
            {
                return Err(format!(
                    "the backward adjacency of relationship table {rel_idx} maps to forward \
                     edge {position} of {}",
                    fwd.num_edges()
                ));
            }
            Some(bwd)
        } else {
            None
        };

        let num_props = read_u32(data, &mut pos)? as usize;
        fits(
            num_props,
            EDGE_COLUMN_MIN_BYTES,
            data,
            pos,
            "edge property columns",
        )?;
        let mut properties: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        for _ in 0..num_props {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);
            let column = read_column(data_bytes, &mut pos, version)
                .map_err(|e| format!("edge codec: {e}"))?;
            properties.insert(key, column);
        }

        let schema = EdgeSchema::new(edge_type.as_str(), rel_table_id);

        let table = RelTable::new(schema, fwd, bwd, properties, src_tid, dst_tid);
        rel_table_id_to_type.push(edge_type);
        rel_tables.push(table);
    }

    let edge_rows: Vec<usize> = rel_tables.iter().map(RelTable::num_edges).collect();

    // The store computes the statistics.
    let mut store = CompactStore::new(node_tables, table_labels, rel_tables, rel_table_id_to_type);

    // ID maps.
    if preserves_ids {
        let (node_id_map, node_offset_to_id) = read_id_map(data, &mut pos, &node_rows)?;
        let (edge_id_map, edge_offset_to_id) = read_id_map(data, &mut pos, &edge_rows)?;
        store.set_id_maps(
            node_id_map,
            edge_id_map,
            node_offset_to_id,
            edge_offset_to_id,
        );
    }

    Ok(store)
}

/// The fewest bytes a node table takes: its label's length, its row count
/// and its column count.
const NODE_TABLE_MIN_BYTES: usize = 2 + 4 + 4;

/// The fewest bytes a node property column takes: its key's length, the zone
/// map flag and the codec's discriminant.
const COLUMN_MIN_BYTES: usize = 2 + 1 + 1;

/// The fewest bytes a relationship table takes: its edge type's length, its
/// two table ids, an empty adjacency (two counts and the edge data flag), the
/// backward flag and the property count.
const REL_TABLE_MIN_BYTES: usize = 2 + 2 + 2 + 9 + 1 + 4;

/// The fewest bytes an edge property column takes: its key's length and the
/// codec's discriminant.
const EDGE_COLUMN_MIN_BYTES: usize = 2 + 1;

/// The bytes of an id map entry: the id, the table and the row.
const ID_MAP_ENTRY_BYTES: usize = 8 + 2 + 8;

/// Refuses more tables than table ids can name.
fn check_table_count(count: usize, what: &str) -> Result<(), String> {
    let most = usize::from(super::id::MAX_TABLE_ID) + 1;
    if count > most {
        return Err(format!(
            "{count} {what}, more than the {most} a compacted base holds"
        ));
    }
    Ok(())
}

/// Refuses `count` items of at least `each` bytes when the bytes left after
/// `pos` cannot hold them, so that nothing is allocated from a count no file
/// could hold.
fn fits(count: usize, each: usize, data: &[u8], pos: usize, what: &str) -> Result<(), String> {
    let left = data.len().saturating_sub(pos);
    if count.checked_mul(each).is_none_or(|needed| needed > left) {
        return Err(format!("{count} {what} do not fit the {left} bytes left"));
    }
    Ok(())
}

/// Refuses an adjacency of relationship table `table` that does not fit the
/// tables it joins: one node per row of the table its edges leave from
/// (`from_rows`), and every target a row of the table they reach (`to_rows`).
fn check_adjacency(
    csr: &CsrAdjacency,
    from_rows: usize,
    to_rows: usize,
    table: usize,
    direction: &str,
) -> Result<(), String> {
    if csr.num_nodes() != from_rows {
        return Err(format!(
            "the {direction} adjacency of relationship table {table} has {} nodes, the table \
             its edges leave from {from_rows} rows",
            csr.num_nodes()
        ));
    }
    if let Some(&target) = csr
        .targets()
        .iter()
        .find(|&&target| target as usize >= to_rows)
    {
        return Err(format!(
            "the {direction} adjacency of relationship table {table} names row {target} of a \
             table of {to_rows} rows"
        ));
    }
    Ok(())
}

/// Reads an id map: its length, then `(id, table, row)` per entry. Every row
/// of every table has exactly one entry (what the writer writes), so the
/// length must be the rows of all tables, each table and row must exist, and
/// no row or id may come twice. The reverse is allocated from the rows, which
/// the length (checked against the bytes left) bounds.
fn read_id_map<Id: MapId>(
    data: &[u8],
    pos: &mut usize,
    rows: &[usize],
) -> Result<(FxHashMap<Id, (u16, u64)>, Vec<Vec<Id>>), String> {
    let what = Id::WHAT;
    let len = read_u32(data, pos)? as usize;
    fits(
        len,
        ID_MAP_ENTRY_BYTES,
        data,
        *pos,
        &format!("{what} id map entries"),
    )?;
    let total = rows
        .iter()
        .try_fold(0usize, |total, &rows| total.checked_add(rows))
        .ok_or_else(|| format!("the {what} tables hold more rows than this platform counts"))?;
    if len != total {
        return Err(format!(
            "the {what} id map holds {len} entries for {total} rows"
        ));
    }
    let mut map = FxHashMap::with_capacity_and_hasher(len, Default::default());
    let mut reverse: Vec<Vec<Id>> = rows.iter().map(|&count| vec![Id::INVALID; count]).collect();
    for _ in 0..len {
        let entry = read_u64(data, pos)?;
        let table = read_u16(data, pos)?;
        let row = read_u64(data, pos)?;
        let entry_id = Id::from_raw(entry);
        if entry_id == Id::INVALID {
            return Err(format!("the {what} id map holds the invalid id {entry}"));
        }
        let tables = reverse.len();
        let slots = reverse.get_mut(usize::from(table)).ok_or_else(|| {
            format!("the {what} id map puts id {entry} in table {table} of {tables}")
        })?;
        let table_rows = slots.len();
        let slot = usize::try_from(row)
            .ok()
            .and_then(|row| slots.get_mut(row))
            .ok_or_else(|| {
                format!(
                    "the {what} id map puts id {entry} at row {row} of table {table}, which has \
                     {table_rows} rows"
                )
            })?;
        if *slot != Id::INVALID {
            return Err(format!(
                "the {what} id map puts ids {} and {entry} both at table {table}, row {row}",
                slot.raw()
            ));
        }
        *slot = entry_id;
        if map.insert(entry_id, (table, row)).is_some() {
            return Err(format!("the {what} id map lists id {entry} twice"));
        }
    }
    Ok((map, reverse))
}

// ── Read helpers ───────────────────────────────────────────────────

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("truncated u16".into());
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let slen = read_u16(data, pos)? as usize;
    if *pos + slen > data.len() {
        return Err("truncated string".into());
    }
    let s =
        std::str::from_utf8(&data[*pos..*pos + slen]).map_err(|_| "invalid UTF-8".to_string())?;
    *pos += slen;
    Ok(s.to_string())
}

/// Steps past a column's zone map: the null and row counts, then the min
/// and the max.
fn skip_zone_map(data: &[u8], pos: &mut usize) -> Result<(), String> {
    read_u32(data, pos)?;
    read_u32(data, pos)?;
    read_optional_value(data, pos)?;
    read_optional_value(data, pos)?;
    Ok(())
}

fn read_optional_value(
    data: &[u8],
    pos: &mut usize,
) -> Result<Option<grafeo_common::types::Value>, String> {
    let tag = *data.get(*pos).ok_or("truncated value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(None),
        1 => {
            // Read raw i64 bytes (written via i64::to_le_bytes).
            if *pos + 8 > data.len() {
                return Err("truncated i64 value".into());
            }
            let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Some(grafeo_common::types::Value::Int64(v)))
        }
        2 => {
            let b = *data.get(*pos).ok_or("truncated bool")?;
            *pos += 1;
            Ok(Some(grafeo_common::types::Value::Bool(b != 0)))
        }
        3 => {
            let s = read_string(data, pos)?;
            Ok(Some(grafeo_common::types::Value::String(
                arcstr::ArcStr::from(s.as_str()),
            )))
        }
        _ => Err(format!("unknown value tag {tag}")),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use grafeo_common::storage::{ChunkMeta, ImageSource, MemoryImage, SectionSink};
    use grafeo_common::types::Value;

    use super::*;

    /// The `CompactStore` section of the database `compact()` wrote with
    /// 0.5.44 (see the fixture's README): encoding 3, ids kept.
    const BASE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.44/compact_store.bin"
    ));

    /// `bytes` as the one raw chunk of a `CompactStore` section, as a 0.5.x
    /// file holds it.
    fn raw_section(bytes: &[u8]) -> MemoryImage {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::CompactStore, 1).unwrap();
        image.write_chunk(ChunkMeta::raw(), bytes).unwrap();
        image
    }

    fn read(bytes: &[u8]) -> grafeo_common::utils::error::Result<CompactStore> {
        let image = raw_section(bytes);
        read_base(&*image.section_source(SectionType::CompactStore).unwrap())
    }

    /// `payload` with its CRC32 appended, as the encoding ends.
    fn with_crc(mut payload: Vec<u8>) -> Vec<u8> {
        let crc = crc32fast::hash(&payload);
        payload.extend_from_slice(&crc.to_le_bytes());
        payload
    }

    /// The fixture without its CRC32, to change and seal again.
    fn payload() -> Vec<u8> {
        BASE[..BASE.len() - 4].to_vec()
    }

    #[test]
    fn the_0_5_44_base_decodes_with_its_ids_labels_and_values() {
        assert_eq!(BASE[4], FORMAT_VERSION_V3, "the fixture is encoding 3");
        let base = read(BASE).unwrap();
        assert!(base.preserves_ids(), "the base keeps its ids");

        let nodes: Vec<_> = base
            .node_ids()
            .into_iter()
            .map(|id| base.get_node(id).unwrap())
            .collect();
        // The first session: Alix, Gus, Vincent, two cities, two documents.
        assert_eq!(nodes.len(), 7);
        let name = |node: &crate::graph::lpg::Node| {
            node.properties
                .get(&PropertyKey::new("name"))
                .or_else(|| node.properties.get(&PropertyKey::new("title")))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let gus = nodes
            .iter()
            .find(|node| name(node).as_deref() == Some("Gus"))
            .unwrap();
        let mut labels: Vec<&str> = gus.labels.iter().map(arcstr::ArcStr::as_str).collect();
        labels.sort_unstable();
        assert_eq!(labels, ["Employee", "Person"], "a 0.5.x key of two labels");
        let alix = nodes
            .iter()
            .find(|node| name(node).as_deref() == Some("Alix"))
            .unwrap();
        assert_eq!(
            alix.properties.get(&PropertyKey::new("tags")),
            Some(&Value::from(r#"["amsterdam", "jazz"]"#)),
            "0.5.44 stored a list as its text"
        );

        let edges: Vec<_> = nodes
            .iter()
            .flat_map(|node| base.outgoing_edges(node.id))
            .collect();
        assert_eq!(edges.len(), 2, "Alix knows Gus and lives in Amsterdam");
        let knows = edges
            .iter()
            .map(|&id| base.get_edge(id).unwrap())
            .find(|edge| edge.edge_type.as_str() == "KNOWS")
            .unwrap();
        assert_eq!((knows.src, knows.dst), (alix.id, gus.id));
        assert_eq!(
            knows.properties.get(&PropertyKey::new("since")),
            Some(&Value::Int64(2019))
        );
    }

    const BASE_0_5_41: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.41/compact_store.bin"
    ));

    /// Properties as `(key, value)`, sorted by key.
    type Properties = Vec<(String, Value)>;

    /// A node as its sorted labels and its properties.
    type NodeContents = (Vec<String>, Properties);

    /// An edge as its type, the names of its source and target, and its
    /// properties.
    type EdgeContents = (String, String, String, Properties);

    /// `properties` sorted by key, leaving out `skip`.
    fn sorted<'a>(
        properties: impl IntoIterator<Item = (&'a PropertyKey, &'a Value)>,
        skip: &[&str],
    ) -> Properties {
        let mut sorted: Properties = properties
            .into_iter()
            .filter(|(key, _)| !skip.contains(&key.as_str()))
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        sorted
    }

    /// The nodes and edges of `base`, each list sorted; `skip` leaves
    /// properties out.
    fn contents(base: &CompactStore, skip: &[&str]) -> (Vec<NodeContents>, Vec<EdgeContents>) {
        let name = |id| {
            let node = base.get_node(id).unwrap();
            ["name", "title"]
                .iter()
                .find_map(|key| node.properties.get(&PropertyKey::new(*key)))
                .and_then(Value::as_str)
                .unwrap()
                .to_string()
        };
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for id in base.node_ids() {
            let node = base.get_node(id).unwrap();
            let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
            labels.sort();
            nodes.push((labels, sorted(&node.properties, skip)));
            for edge_id in base.outgoing_edges(id) {
                let edge = base.get_edge(edge_id).unwrap();
                edges.push((
                    edge.edge_type.to_string(),
                    name(edge.src),
                    name(edge.dst),
                    sorted(&edge.properties, skip),
                ));
            }
        }
        nodes.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        edges.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        (nodes, edges)
    }

    /// 0.5.40 and 0.5.41 wrote encoding 1, with every column whole: the same
    /// first session decodes to the same nodes, edges and values as the
    /// 0.5.44 base, but for the temporal values 0.5.41 could not write.
    #[test]
    fn the_0_5_41_base_decodes_like_the_0_5_44_one() {
        assert_eq!(
            BASE_0_5_41[4], FORMAT_VERSION_V1,
            "the fixture is encoding 1"
        );
        let v1 = read(BASE_0_5_41).unwrap();
        assert!(v1.preserves_ids(), "the base keeps its ids");
        let temporal = ["born", "seen", "stay"];
        let (nodes, edges) = contents(&v1, &temporal);
        assert_eq!(nodes.len(), 7);
        assert_eq!(edges.len(), 2);
        assert_eq!((nodes, edges), contents(&read(BASE).unwrap(), &temporal));
    }

    #[test]
    fn a_damaged_base_is_refused() {
        let mut flipped = BASE.to_vec();
        flipped[20] ^= 0xFF;
        let error = read(&flipped).map(drop).unwrap_err().to_string();
        assert!(
            error.contains("CompactStore") && error.contains("CRC32"),
            "{error}"
        );

        let error = read(&BASE[..9]).map(drop).unwrap_err().to_string();
        assert!(error.contains("too short"), "{error}");

        let mut magic = payload();
        magic[0] = b'X';
        let error = read(&with_crc(magic)).map(drop).unwrap_err().to_string();
        assert!(error.contains("bad magic"), "{error}");
    }

    #[test]
    fn an_encoding_no_0_5_release_wrote_is_refused() {
        // 2: blocks without zone maps, between 0.5.41 and 0.5.42; 4 and 5:
        // 0.6.0 development builds.
        for version in [0u8, 2, 4, 5] {
            let mut bytes = payload();
            bytes[4] = version;
            let error = read(&with_crc(bytes)).map(drop).unwrap_err().to_string();
            assert!(
                error.contains(&format!(
                    "unsupported CompactStore section version {version}"
                )),
                "{version}: {error}"
            );
        }
    }

    /// Counts are checked against the bytes left before anything is
    /// allocated for them, also when the CRC32 matches.
    #[test]
    fn a_count_no_file_could_hold_is_refused_before_allocating() {
        let mut bytes = payload();
        // The node table count follows the magic, the version and the flags.
        bytes[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = read(&with_crc(bytes)).map(drop).unwrap_err().to_string();
        assert!(error.contains("node tables"), "{error}");

        let mut bytes = payload();
        bytes[6..10].copy_from_slice(&40_000u32.to_le_bytes());
        let error = read(&with_crc(bytes)).map(drop).unwrap_err().to_string();
        assert!(error.contains("node tables"), "{error}");
    }

    /// A section that is not one raw chunk was written by a 0.6.0
    /// development build: refused with what to do.
    #[test]
    fn a_chunked_section_is_refused() {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::CompactStore, 5).unwrap();
        image.write_chunk(ChunkMeta::meta(), &[1, 0]).unwrap();
        let error = read_base(&*image.section_source(SectionType::CompactStore).unwrap())
            .map(drop)
            .unwrap_err()
            .to_string();
        assert!(error.contains("development build"), "{error}");
    }
}
