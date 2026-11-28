//! The records durable change data capture writes (#679 patch 6): kind 2, a group begin with the
//! group's first change timestamp, and kind 26, a write with what it replaced. Both are written
//! required; replay applies kind 26's op as it applies the plain data kind. A group begin without
//! a clock is kind 1, byte for byte as before.

use std::sync::Arc;

use super::{
    BeforeImage, GroupBegin, LogRecord, MAX_RECORD_BYTES, Origin, RECORD_HEADER_BYTES,
    read_log_records,
};
use crate::change::{DataOp, Labels};
use crate::types::{ArcStr, EdgeId, EpochId, HlcTimestamp, NodeId, Value};
use crate::utils::error::{Error, Result};

fn framed(record: &LogRecord) -> Vec<u8> {
    let mut bytes = Vec::new();
    record.encode_framed(&mut bytes).unwrap();
    bytes
}

fn read_all(bytes: &[u8]) -> Result<Vec<LogRecord>> {
    let mut records = Vec::new();
    read_log_records(bytes, &mut |record| {
        records.push(record);
        Ok(())
    })?;
    Ok(records)
}

fn read_error(bytes: &[u8]) -> String {
    match read_all(bytes) {
        Ok(records) => panic!("{} records read, expected an error", records.len()),
        Err(error) => error.to_string(),
    }
}

fn labels(items: &[&str]) -> Labels {
    items.iter().map(|&item| ArcStr::from(item)).collect()
}

fn with_before(graph: Option<&str>, op: DataOp, before: BeforeImage) -> LogRecord {
    LogRecord::DataWithBefore {
        graph: graph.map(str::to_string),
        op,
        before,
    }
}

fn begin(clock: Option<HlcTimestamp>) -> LogRecord {
    LogRecord::GroupBegin(GroupBegin {
        epoch: EpochId::new(88),
        timestamp_ms: 1_696_500_000_123,
        origin: Origin::Transaction,
        clock,
    })
}

/// A set of Paris over Amsterdam, as kind 26.
fn paris_over_amsterdam() -> LogRecord {
    with_before(
        None,
        DataOp::SetNodeProperty {
            id: NodeId::new(3),
            key: "city".into(),
            value: Value::from("Paris"),
        },
        BeforeImage::Property {
            old: Some(Value::from("Amsterdam")),
        },
    )
}

/// The bytes of kinds 2 and 26, pinned. Kind 2 is kind 1's payload with the group's first change
/// timestamp after it (its raw `u64`); kind 26 is the inner op's kind, the inner record's payload
/// as that kind writes it, then the image (its variant index first, a value as the value codec's
/// bytes).
#[test]
fn kinds_2_and_26_are_pinned() {
    #[rustfmt::skip]
    let clocked: &[u8] = &[
        2, 1, 20, 0, 0, 0, // kind 2 (GroupBegin with a clock), required, 20 bytes
        88, // epoch 88
        253, 123, 165, 71, 255, 138, 1, 0, 0, // timestamp 1,696,500,000,123 ms (a u64)
        0, // origin Transaction
        253, 3, 0, 123, 165, 71, 255, 138, 1, // the first change: 1,696,500,000,123 ms, logical 3
    ];
    assert_eq!(
        framed(&begin(Some(HlcTimestamp::new(1_696_500_000_123, 3)))),
        clocked
    );
    #[rustfmt::skip]
    let replaced: &[u8] = &[
        26, 1, 36, 0, 0, 0, // kind 26 (DataWithBefore), required, 36 bytes
        20, // the inner kind: 20 (SetNodeProperty)
        0, 3, // the default graph, id 3
        4, 99, 105, 116, 121, // "city"
        10, 4, 5, 0, 0, 0, 80, 97, 114, 105, 115, // the value codec's "Paris", 10 bytes
        0, // the image: Property
        1, // Some
        14, 4, 9, 0, 0, 0, 65, 109, 115, 116, 101, 114, 100, 97, 109, // "Amsterdam", 14 bytes
    ];
    assert_eq!(framed(&paris_over_amsterdam()), replaced);
    for bytes in [clocked, replaced] {
        let back = read_all(bytes).unwrap();
        assert_eq!(framed(&back[0]), bytes, "decodes to what encodes the same");
    }
}

/// A begin without a clock is kind 1, its bytes as today's; one with a clock is kind 2, and each
/// decodes to `GroupBegin` with the clock it had.
#[test]
fn a_begin_is_kind_1_without_a_clock_and_kind_2_with_one() {
    #[rustfmt::skip]
    let plain: &[u8] = &[
        1, 1, 11, 0, 0, 0,
        88,
        253, 123, 165, 71, 255, 138, 1, 0, 0,
        0,
    ];
    assert_eq!(framed(&begin(None)), plain);
    for clock in [None, Some(HlcTimestamp::new(1_696_500_000_123, 65_535))] {
        let record = begin(clock);
        assert_eq!(record.kind(), if clock.is_some() { 2 } else { 1 });
        assert_eq!(read_all(&framed(&record)).unwrap(), [record]);
    }
}

/// Every write with a before-image round-trips with its image, in the default graph and a named
/// one, over values of every kind the codec keeps (NaN payloads, -0.0, offsets).
#[test]
fn every_inner_kind_round_trips_with_its_image() {
    let values = [
        Value::Null,
        Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058)),
        Value::Float64(-0.0),
        Value::from("Barcelona"),
        Value::List(Arc::from(vec![Value::Int64(3), Value::from("Berlin")])),
    ];
    let properties: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, value)| (format!("p{i}").as_str().into(), value.clone()))
        .collect();
    let mut records = Vec::new();
    for graph in [None, Some("trips")] {
        records.push(with_before(
            graph,
            DataOp::DeleteNode {
                id: NodeId::new(19),
            },
            BeforeImage::Node {
                labels: labels(&["Person", "Pilot"]),
                properties: properties.clone(),
            },
        ));
        for deleted_with in [None, Some(NodeId::new(19))] {
            records.push(with_before(
                graph,
                DataOp::DeleteEdge {
                    id: EdgeId::new(88),
                },
                BeforeImage::Edge {
                    edge_type: "KNOWS".into(),
                    src: NodeId::new(3),
                    dst: NodeId::new(19),
                    properties: properties.clone(),
                    deleted_with,
                },
            ));
        }
        for old in values.iter().cloned().map(Some).chain([None]) {
            records.push(with_before(
                graph,
                DataOp::SetNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                    value: Value::from("Paris"),
                },
                BeforeImage::Property { old: old.clone() },
            ));
            records.push(with_before(
                graph,
                DataOp::SetEdgeProperty {
                    id: EdgeId::new(88),
                    key: "km".into(),
                    value: Value::Int64(319),
                },
                BeforeImage::Property { old },
            ));
        }
        for old in &values {
            records.push(with_before(
                graph,
                DataOp::RemoveNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                },
                BeforeImage::Property {
                    old: Some(old.clone()),
                },
            ));
            records.push(with_before(
                graph,
                DataOp::RemoveEdgeProperty {
                    id: EdgeId::new(88),
                    key: "km".into(),
                },
                BeforeImage::Property {
                    old: Some(old.clone()),
                },
            ));
        }
        records.push(with_before(
            graph,
            DataOp::AddNodeLabel {
                id: NodeId::new(19),
                label: "Pilot".into(),
            },
            BeforeImage::Labels {
                old: labels(&["Person"]),
            },
        ));
        records.push(with_before(
            graph,
            DataOp::RemoveNodeLabel {
                id: NodeId::new(19),
                label: "Pilot".into(),
            },
            BeforeImage::Labels {
                old: labels(&["Person", "Pilot"]),
            },
        ));
    }
    let mut kinds = Vec::new();
    for record in &records {
        let bytes = framed(record);
        assert_eq!(bytes[0], 26, "{record:?}");
        kinds.push(bytes[RECORD_HEADER_BYTES]);
        let back = read_all(&bytes).unwrap();
        assert_eq!(framed(&back[0]), bytes, "{record:?}");
    }
    kinds.sort_unstable();
    kinds.dedup();
    assert_eq!(kinds, [17, 19, 20, 21, 22, 23, 24, 25], "every inner kind");
}

/// Kind 26 holds a write with what it replaced: a create or a triple inside it is refused when
/// read, naming kind 26 and the inner kind.
#[test]
fn a_create_or_a_triple_inside_kind_26_is_refused() {
    let bytes = framed(&paris_over_amsterdam());
    for inner in [16_u8, 18, 64, 65, 1, 26, 99] {
        let mut forged = bytes.clone();
        forged[RECORD_HEADER_BYTES] = inner;
        let error = read_error(&forged);
        assert!(
            error.contains("kind 26") && error.contains(&inner.to_string()),
            "inner kind {inner}: {error}"
        );
    }
}

/// An image that does not fit its write (labels for a property set, a property for a delete) is
/// refused: when written, and when read from bytes that were forged.
#[test]
fn an_image_that_does_not_fit_its_write_is_refused() {
    let mismatched = [
        with_before(
            None,
            DataOp::SetNodeProperty {
                id: NodeId::new(3),
                key: "city".into(),
                value: Value::from("Paris"),
            },
            BeforeImage::Labels {
                old: labels(&["Person"]),
            },
        ),
        with_before(
            None,
            DataOp::DeleteNode { id: NodeId::new(3) },
            BeforeImage::Property { old: None },
        ),
        with_before(
            None,
            DataOp::RemoveNodeProperty {
                id: NodeId::new(3),
                key: "city".into(),
            },
            BeforeImage::Property { old: None },
        ),
        with_before(
            None,
            DataOp::AddNodeLabel {
                id: NodeId::new(3),
                label: "Pilot".into(),
            },
            BeforeImage::Labels {
                old: labels(&["Pilot"]),
            },
        ),
    ];
    for record in mismatched {
        let mut out = vec![3, 19, 88];
        let error = record.encode_framed(&mut out).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
    }
    // The pinned Paris record with its image's variant turned from Property (0) into Labels (1):
    // the rest decodes as a list of one label, which does not fit a property set.
    let mut forged = framed(&paris_over_amsterdam());
    let image = RECORD_HEADER_BYTES + 1 + 18;
    assert_eq!(forged[image], 0);
    forged[image] = 1;
    let error = read_error(&forged);
    assert!(error.contains("kind 26"), "{error}");
}

/// An image counts toward the record's maximum: a write whose image would take the framed record
/// past it is refused from the sizes, before a byte is written.
#[test]
#[cfg_attr(
    miri,
    ignore = "a list of 2^20 values takes minutes under Miri; this module has no unsafe code"
)]
fn an_image_over_the_maximum_is_refused_before_it_is_written() {
    let line = Value::from("Amsterdam ".repeat(103).as_str());
    let lines = Value::List(Arc::from(vec![line; 1 << 20]));
    let record = with_before(
        None,
        DataOp::SetNodeProperty {
            id: NodeId::new(3),
            key: "notes".into(),
            value: Value::Int64(1),
        },
        BeforeImage::Property { old: Some(lines) },
    );
    let mut out = vec![3, 19, 88];
    let error = record.encode_framed(&mut out).unwrap_err();
    assert!(matches!(error, Error::Serialization(_)), "{error:?}");
    assert!(
        error.to_string().contains("kind 26")
            && error
                .to_string()
                .contains(&(MAX_RECORD_BYTES - RECORD_HEADER_BYTES).to_string()),
        "{error}"
    );
    assert_eq!(out, [3, 19, 88]);
    assert!(out.capacity() < 1 << 20, "the image was never encoded");
}

/// An image's labels and properties keep the list rule of an op's: a deleted node with more
/// labels than a reader decodes is refused when written, before a byte of it.
#[test]
fn an_image_list_past_the_rule_is_refused_when_written() {
    let record = with_before(
        None,
        DataOp::DeleteNode { id: NodeId::new(3) },
        BeforeImage::Node {
            labels: (0..200_000).map(|_| ArcStr::from("")).collect(),
            properties: Vec::new(),
        },
    );
    let mut out = vec![3, 19, 88];
    let error = record.encode_framed(&mut out).unwrap_err();
    assert!(
        matches!(&error, Error::Serialization(message) if message.contains("labels")),
        "{error:?}"
    );
    assert_eq!(out, [3, 19, 88]);
}

/// The bytes of the Labels, Node and Edge images, and of the identities a delete logs when its
/// properties are left out, pinned field by field: a round-trip alone would pass with two fields
/// swapped (an edge's source and target).
#[test]
fn the_labels_node_and_edge_images_are_pinned() {
    let cases: [(LogRecord, &[u8]); 5] = [
        (
            with_before(
                None,
                DataOp::AddNodeLabel {
                    id: NodeId::new(3),
                    label: "Pilot".into(),
                },
                BeforeImage::Labels {
                    old: labels(&["Person"]),
                },
            ),
            &[
                26, 1, 18, 0, 0, 0, 24, 0, 3, 5, 80, 105, 108, 111, 116, // kind 24's payload
                1, 1, 6, 80, 101, 114, 115, 111, 110, // Labels: one label, "Person"
            ],
        ),
        (
            with_before(
                None,
                DataOp::DeleteNode {
                    id: NodeId::new(19),
                },
                BeforeImage::Node {
                    labels: labels(&["Person"]),
                    properties: Vec::new(),
                },
            ),
            &[
                26, 1, 13, 0, 0, 0, 17, 0, 19, // kind 17's payload
                2, 1, 6, 80, 101, 114, 115, 111, 110, 0, // Node: "Person", no properties
            ],
        ),
        (
            with_before(
                None,
                DataOp::DeleteEdge {
                    id: EdgeId::new(88),
                },
                BeforeImage::Edge {
                    edge_type: "KNOWS".into(),
                    src: NodeId::new(3),
                    dst: NodeId::new(19),
                    properties: Vec::new(),
                    deleted_with: Some(NodeId::new(19)),
                },
            ),
            &[
                26, 1, 15, 0, 0, 0, 19, 0, 88, // kind 19's payload
                3, 5, 75, 78, 79, 87, 83, 3, 19, 0, 1,
                19, // Edge: KNOWS, 3 to 19, none, with 19
            ],
        ),
        (
            with_before(
                None,
                DataOp::DeleteNode {
                    id: NodeId::new(19),
                },
                BeforeImage::NodeIdentity {
                    labels: labels(&["Person"]),
                },
            ),
            &[
                26, 1, 12, 0, 0, 0, 17, 0, 19, // kind 17's payload
                4, 1, 6, 80, 101, 114, 115, 111, 110, // the node's identity: "Person"
            ],
        ),
        (
            with_before(
                None,
                DataOp::DeleteEdge {
                    id: EdgeId::new(88),
                },
                BeforeImage::EdgeIdentity {
                    edge_type: "KNOWS".into(),
                    src: NodeId::new(3),
                    dst: NodeId::new(19),
                    deleted_with: None,
                },
            ),
            &[
                26, 1, 13, 0, 0, 0, 19, 0, 88, // kind 19's payload
                5, 5, 75, 78, 79, 87, 83, 3, 19, 0, // the edge's identity: KNOWS, 3 to 19
            ],
        ),
    ];
    for (record, bytes) in cases {
        assert_eq!(framed(&record), bytes, "{record:?}");
        assert_eq!(read_all(bytes).unwrap(), [record]);
    }
}

/// An image variant this release does not know, and bytes after the image, are refused when
/// read, naming kind 26.
#[test]
fn an_unknown_image_variant_and_trailing_bytes_are_refused() {
    let bytes = framed(&paris_over_amsterdam());
    let image = RECORD_HEADER_BYTES + 1 + 18;
    let mut unknown = bytes.clone();
    unknown[image] = 6;
    assert!(read_error(&unknown).contains("kind 26"));
    let mut trailing = bytes;
    trailing.push(0);
    trailing[2] += 1;
    assert!(read_error(&trailing).contains("kind 26"));
}

/// One rule says what of a write's before-image is logged: the whole image when the record
/// keeps the record limits with it; for a delete whose properties pass the record's byte maximum
/// or the list rule, its identity alone (its labels, or its type and endpoints); nothing when the
/// identity itself passes them (labels past the list rule) or the image is no delete's. A fitting
/// property image is logged whole; a create has none.
#[test]
#[cfg_attr(
    miri,
    ignore = "a list of 2^20 values takes minutes under Miri; this module has no unsafe code"
)]
fn the_rule_for_a_logged_image_follows_the_record_limits() {
    use super::{ImageLogged, logs_before_image};
    use crate::change::{Before, EdgeImage, NodeImage};
    let delete = DataOp::DeleteNode { id: NodeId::new(3) };
    let node =
        |labels: Labels, properties| Before::Node(Box::new(NodeImage { labels, properties }));
    let person = || labels(&["Person"]);
    let line = Value::from("Amsterdam ".repeat(103).as_str());
    let notes = Value::List(Arc::from(vec![line; 1 << 20]));
    let huge = vec![("notes".into(), notes.clone())];
    assert_eq!(
        logs_before_image(None, &delete, &node(person(), huge.clone())),
        ImageLogged::Identity,
        "past the byte maximum"
    );
    // Three-letter keys and nulls: each property takes fewer bytes than its slot allows for.
    let many: Vec<_> = (0..100_000)
        .map(|i| (tiny_key(i).as_str().into(), Value::Null))
        .collect();
    assert_eq!(
        logs_before_image(None, &delete, &node(person(), many)),
        ImageLogged::Identity,
        "past the list rule"
    );
    let small = vec![("city".into(), Value::from("Paris"))];
    assert_eq!(
        logs_before_image(None, &delete, &node(person(), small)),
        ImageLogged::Whole,
        "a small image"
    );
    assert_eq!(
        logs_before_image(None, &delete, &node(person(), Vec::new())),
        ImageLogged::Whole,
        "an empty one"
    );
    // Empty labels take fewer bytes than their slots: 200,000 of them pass the list rule, so
    // even the identity is not logged.
    let crowded: Labels = (0..200_000).map(|_| ArcStr::from("")).collect();
    assert_eq!(
        logs_before_image(None, &delete, &node(crowded, huge.clone())),
        ImageLogged::Nothing,
        "an identity past the list rule"
    );
    let edge = Before::Edge(Box::new(EdgeImage {
        src: NodeId::new(3),
        dst: NodeId::new(19),
        edge_type: "KNOWS".into(),
        properties: huge,
        deleted_with: None,
    }));
    assert_eq!(
        logs_before_image(
            None,
            &DataOp::DeleteEdge {
                id: EdgeId::new(88)
            },
            &edge
        ),
        ImageLogged::Identity,
        "an edge past the byte maximum"
    );
    let set = DataOp::SetNodeProperty {
        id: NodeId::new(3),
        key: "city".into(),
        value: Value::from("Paris"),
    };
    assert_eq!(
        logs_before_image(None, &set, &Before::Value(None)),
        ImageLogged::Whole
    );
    assert_eq!(
        logs_before_image(None, &set, &Before::Value(Some(notes))),
        ImageLogged::Nothing,
        "an old value past the byte maximum has no identity to keep"
    );
    assert_eq!(
        logs_before_image(None, &set, &Before::Absent),
        ImageLogged::Nothing
    );
    assert_eq!(
        logs_before_image(None, &delete, &Before::Value(None)),
        ImageLogged::Nothing,
        "a misfit"
    );
}

/// A distinct three-letter key for `i` below 62^3.
fn tiny_key(i: usize) -> String {
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    [i / 3844, i / 62 % 62, i % 62]
        .iter()
        .map(|&digit| char::from(LETTERS[digit]))
        .collect()
}

/// A record's image as the change set holds it: a delete's identity alone (variants 4 and 5)
/// gives the node or the edge with no properties and the flag that they were left out; every
/// other image is whole.
#[test]
fn an_identity_image_reads_as_left_out_properties() {
    use super::ImageLogged;
    use crate::change::Before;
    let (node, logged) = BeforeImage::NodeIdentity {
        labels: labels(&["Person"]),
    }
    .into_logged();
    assert_eq!(logged, ImageLogged::Identity);
    assert!(
        matches!(&node, Before::Node(image) if image.properties.is_empty()
            && image.labels.as_slice() == ["Person"]),
        "{node:?}"
    );
    let (edge, logged) = BeforeImage::EdgeIdentity {
        edge_type: "KNOWS".into(),
        src: NodeId::new(3),
        dst: NodeId::new(19),
        deleted_with: None,
    }
    .into_logged();
    assert_eq!(logged, ImageLogged::Identity);
    assert!(
        matches!(&edge, Before::Edge(image) if image.properties.is_empty()
            && image.src == NodeId::new(3) && image.dst == NodeId::new(19)),
        "{edge:?}"
    );
    for whole in [
        BeforeImage::Property { old: None },
        BeforeImage::Labels {
            old: labels(&["Person"]),
        },
        BeforeImage::Node {
            labels: labels(&["Person"]),
            properties: Vec::new(),
        },
        BeforeImage::Edge {
            edge_type: "KNOWS".into(),
            src: NodeId::new(3),
            dst: NodeId::new(19),
            properties: Vec::new(),
            deleted_with: None,
        },
    ] {
        assert_eq!(
            whole.clone().into_logged().1,
            ImageLogged::Whole,
            "{whole:?}"
        );
    }
}
