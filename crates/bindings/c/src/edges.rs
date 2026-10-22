//! A node's edges without a query (#402): `(neighbor, edge)` pairs out as two
//! typed id arrays of the same length, each freed with
//! `grafeo_free_node_ids`, not as JSON.

use std::ffi::c_char;

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_engine::Direction;

use crate::batch::{invalid, json_arg, labels};
use crate::error::{GrafeoStatus, set_last_error};
use crate::types::GrafeoDatabase;

/// `GRAFEO_DIRECTION_OUT`: the edges that leave the node.
pub(crate) const DIRECTION_OUT: i32 = 0;
/// `GRAFEO_DIRECTION_IN`: the edges that reach the node.
pub(crate) const DIRECTION_IN: i32 = 1;
/// `GRAFEO_DIRECTION_BOTH`: both, a self-loop once.
pub(crate) const DIRECTION_BOTH: i32 = 2;

/// The engine direction of a `GrafeoDirection` value.
pub(crate) fn direction(value: i32) -> Result<Direction, GrafeoStatus> {
    match value {
        DIRECTION_OUT => Ok(Direction::Outgoing),
        DIRECTION_IN => Ok(Direction::Incoming),
        DIRECTION_BOTH => Ok(Direction::Both),
        other => Err(invalid(&format!(
            "direction: {other} is not GRAFEO_DIRECTION_OUT (0), _IN (1) or _BOTH (2)"
        ))),
    }
}

/// The edge types of `edge_types_json`: NULL for every type, else a JSON
/// array of strings.
pub(crate) fn edge_types(edge_types_json: *const c_char) -> Result<Vec<String>, GrafeoStatus> {
    if edge_types_json.is_null() {
        return Ok(Vec::new());
    }
    labels(
        &json_arg(edge_types_json, "edge_types_json")?,
        "edge_types_json",
    )
}

/// Zeroes the three outputs, so no error path leaves them uninitialized; an
/// error if one is null.
pub(crate) fn clear_edges(
    out_neighbors: *mut *mut u64,
    out_edges: *mut *mut u64,
    out_count: *mut usize,
) -> Result<(), GrafeoStatus> {
    if out_neighbors.is_null() || out_edges.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return Err(GrafeoStatus::ErrorNullPointer);
    }
    // SAFETY: all three checked not null above.
    unsafe {
        *out_neighbors = std::ptr::null_mut();
        *out_edges = std::ptr::null_mut();
        *out_count = 0;
    }
    Ok(())
}

/// An id array for the caller, freed with `grafeo_free_node_ids(ptr, len)`
/// (a boxed slice, so its allocation is exactly `len` ids); null when empty.
fn id_array(ids: Vec<u64>) -> *mut u64 {
    if ids.is_empty() {
        return std::ptr::null_mut();
    }
    Box::into_raw(ids.into_boxed_slice()).cast::<u64>()
}

/// Hands the pairs to the caller as two arrays of the same length, through
/// outputs `clear_edges` checked.
pub(crate) fn write_edges(
    edges: &[(NodeId, EdgeId)],
    out_neighbors: *mut *mut u64,
    out_edges: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let neighbors = id_array(edges.iter().map(|(node, _)| node.as_u64()).collect());
    let edge_ids = id_array(edges.iter().map(|(_, edge)| edge.as_u64()).collect());
    // SAFETY: `clear_edges` checked all three are not null.
    unsafe {
        *out_neighbors = neighbors;
        *out_edges = edge_ids;
        *out_count = edges.len();
    }
    GrafeoStatus::Ok
}

/// The direction and edge types of a call, after clearing its outputs.
pub(crate) fn edges_args(
    direction_value: i32,
    edge_types_json: *const c_char,
    out_neighbors: *mut *mut u64,
    out_edges: *mut *mut u64,
    out_count: *mut usize,
) -> Result<(Direction, Vec<String>), GrafeoStatus> {
    clear_edges(out_neighbors, out_edges, out_count)?;
    Ok((direction(direction_value)?, edge_types(edge_types_json)?))
}

/// The edges of node `node_id` in `direction` (`GRAFEO_DIRECTION_OUT`, `_IN`
/// or `_BOTH`, which lists a self-loop once), only those of the types in
/// `edge_types_json` (NULL for every type, else a JSON array of strings), as
/// committed. Neighbor `i` is reached through edge `i`: `*out_neighbors` and
/// `*out_edges` hold `*out_count` ids each (NULL when there are none), each
/// freed with `grafeo_free_node_ids(ptr, *out_count)`. A node that does not
/// exist has no edges.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_get_node_edges(
    db: *mut GrafeoDatabase,
    node_id: u64,
    direction: i32,
    edge_types_json: *const c_char,
    out_neighbors: *mut *mut u64,
    out_edges: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let (direction, edge_types) = match edges_args(
        direction,
        edge_types_json,
        out_neighbors,
        out_edges,
        out_count,
    ) {
        Ok(args) => args,
        Err(status) => return status,
    };
    let edge_types: Vec<&str> = edge_types.iter().map(String::as_str).collect();
    let edges = match db
        .inner
        .read()
        .get_node_edges(NodeId::new(node_id), direction, &edge_types)
    {
        Ok(edges) => edges,
        Err(err) => return crate::error::set_error(&err),
    };
    write_edges(&edges, out_neighbors, out_edges, out_count)
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use super::*;
    use crate::database::{
        grafeo_close, grafeo_create_edge, grafeo_create_node, grafeo_delete_edge,
        grafeo_free_database, grafeo_free_node_ids, grafeo_open_memory,
    };
    use crate::error::grafeo_last_error;

    fn c(text: &str) -> CString {
        CString::new(text).unwrap()
    }

    fn node(db: *mut GrafeoDatabase) -> u64 {
        let labels = c("[\"Person\"]");
        grafeo_create_node(db, labels.as_ptr(), std::ptr::null())
    }

    fn edge(db: *mut GrafeoDatabase, src: u64, dst: u64, edge_type: &str) -> u64 {
        let edge_type = c(edge_type);
        grafeo_create_edge(db, src, dst, edge_type.as_ptr(), std::ptr::null())
    }

    /// The call's pairs, sorted by edge id, with both arrays freed.
    fn pairs(
        db: *mut GrafeoDatabase,
        node: u64,
        direction: i32,
        edge_types: Option<&str>,
    ) -> Vec<(u64, u64)> {
        let edge_types = edge_types.map(c);
        let mut neighbors = std::ptr::null_mut();
        let mut edges = std::ptr::null_mut();
        let mut count = 0;
        let status = grafeo_get_node_edges(
            db,
            node,
            direction,
            edge_types.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
            &raw mut neighbors,
            &raw mut edges,
            &raw mut count,
        );
        assert_eq!(status, GrafeoStatus::Ok);
        let mut result = if count == 0 {
            assert!(neighbors.is_null() && edges.is_null());
            Vec::new()
        } else {
            // SAFETY: both arrays hold `count` ids.
            let (n, e) = unsafe {
                (
                    std::slice::from_raw_parts(neighbors, count),
                    std::slice::from_raw_parts(edges, count),
                )
            };
            n.iter().copied().zip(e.iter().copied()).collect()
        };
        grafeo_free_node_ids(neighbors, count);
        grafeo_free_node_ids(edges, count);
        result.sort_by_key(|&(_, edge)| edge);
        result
    }

    #[test]
    fn edges_by_direction_and_type() {
        let db = grafeo_open_memory();
        let (alix, gus, vincent) = (node(db), node(db), node(db));
        let knows_gus = edge(db, alix, gus, "KNOWS");
        let knows_vincent = edge(db, alix, vincent, "KNOWS");
        let works_with = edge(db, gus, alix, "WORKS_WITH");
        let note = edge(db, alix, alix, "NOTE");

        assert_eq!(
            pairs(db, alix, DIRECTION_OUT, None),
            vec![(gus, knows_gus), (vincent, knows_vincent), (alix, note)]
        );
        assert_eq!(
            pairs(db, alix, DIRECTION_IN, None),
            vec![(gus, works_with), (alix, note)]
        );
        assert_eq!(
            pairs(db, alix, DIRECTION_BOTH, None),
            vec![
                (gus, knows_gus),
                (vincent, knows_vincent),
                (gus, works_with),
                (alix, note)
            ]
        );
        assert_eq!(
            pairs(db, alix, DIRECTION_BOTH, Some("[\"KNOWS\", \"NOTE\"]")),
            vec![(gus, knows_gus), (vincent, knows_vincent), (alix, note)]
        );

        assert_eq!(grafeo_delete_edge(db, knows_gus), 1);
        assert_eq!(pairs(db, gus, DIRECTION_IN, None), Vec::<(u64, u64)>::new());
        assert_eq!(
            pairs(db, 88_888, DIRECTION_BOTH, None),
            Vec::<(u64, u64)>::new()
        );
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn bad_arguments_fail_and_clear_the_outputs() {
        let db = grafeo_open_memory();
        let alix = node(db);
        let mut neighbors = std::ptr::dangling_mut::<u64>();
        let mut edges = std::ptr::dangling_mut::<u64>();
        let mut count = 7;
        let status = grafeo_get_node_edges(
            db,
            alix,
            3,
            std::ptr::null(),
            &raw mut neighbors,
            &raw mut edges,
            &raw mut count,
        );
        assert_eq!(status, GrafeoStatus::ErrorSerialization);
        assert!(neighbors.is_null() && edges.is_null() && count == 0);
        // SAFETY: the error string lives until the next call on this thread.
        let message = unsafe { CStr::from_ptr(grafeo_last_error()) }.to_string_lossy();
        assert!(message.contains("direction"), "{message}");

        let not_a_list = c("\"KNOWS\"");
        let status = grafeo_get_node_edges(
            db,
            alix,
            DIRECTION_OUT,
            not_a_list.as_ptr(),
            &raw mut neighbors,
            &raw mut edges,
            &raw mut count,
        );
        assert_eq!(status, GrafeoStatus::ErrorSerialization);
        grafeo_close(db);
        grafeo_free_database(db);
    }
}
