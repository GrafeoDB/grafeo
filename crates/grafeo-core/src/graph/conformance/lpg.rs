//! `LpgStore` against the conformance suite, with its known gaps per build.

use super::{KnownGap, conformance_suite};
use crate::graph::lpg::LpgStore;

/// A store with the property index on `name` the suite checks.
fn lpg_store() -> LpgStore {
    let store = LpgStore::new().expect("a new store");
    store.create_property_index("name");
    store
}

/// An open transaction's values and labels are written in place (#412).
const IN_PLACE: &str = "#412: values and labels are written in place, seen by every reader at once";

/// A delete is marked at the deleting transaction's start epoch (inbox
/// `transactions/delete-marked-at-the-start-epoch.md`).
const DELETE_AT_START: &str =
    "deletes are marked at the deleting transaction's start epoch, not at its commit";

/// The cases `LpgStore` fails, each with its reason. The row-group store
/// passes all of them from H2 on.
const GAPS: &[KnownGap] = &[
    KnownGap {
        case: "an_open_transactions_value_and_label_writes_are_invisible_to_others",
        reason: IN_PLACE,
    },
    KnownGap {
        case: "a_reader_at_an_earlier_epoch_keeps_its_values_and_labels",
        reason: IN_PLACE,
    },
    KnownGap {
        case: "an_open_transactions_deletes_are_invisible_to_others",
        reason: DELETE_AT_START,
    },
    KnownGap {
        case: "a_reader_at_an_earlier_epoch_keeps_deleted_entities",
        reason: DELETE_AT_START,
    },
    KnownGap {
        case: "a_delete_is_marked_at_its_commit_epoch",
        reason: DELETE_AT_START,
    },
    KnownGap {
        case: "counts_move_at_commit_only",
        reason: "the counts read the labels written in place and the deletes marked at the start epoch",
    },
];

conformance_suite!(LpgStore, lpg_store, GAPS);
