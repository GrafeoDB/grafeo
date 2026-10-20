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
    KnownGap {
        case: "every_read_path_hides_an_open_transactions_values_and_labels",
        reason: IN_PLACE,
    },
    KnownGap {
        case: "every_read_path_at_an_earlier_epoch_keeps_values_and_labels",
        reason: IN_PLACE,
    },
    KnownGap {
        case: "an_immediate_write_keeps_the_old_value_for_an_older_reader",
        reason: IN_PLACE,
    },
    KnownGap {
        case: "every_read_path_hides_an_open_transactions_creates_and_deletes",
        reason: DELETE_AT_START,
    },
    KnownGap {
        case: "every_read_path_at_an_earlier_epoch_keeps_creates_and_deletes",
        reason: DELETE_AT_START,
    },
    KnownGap {
        case: "every_read_path_sees_the_transactions_own_writes",
        reason: "a lookup of a key without an index and a range lookup scan the nodes \
                 committed now, without the transaction's own creates: the reads at a view \
                 keep LpgStore's lookups of now (#412 decision Q2, no interim fix)",
    },
];

/// The known gaps of this build: [`GAPS`], and without `temporal` a value
/// set to null, which the property storage keeps as a stored null (the
/// version logs of `temporal` read it as no value).
fn gaps() -> Vec<KnownGap> {
    let mut gaps = GAPS.to_vec();
    if !cfg!(feature = "temporal") {
        gaps.push(KnownGap {
            case: "a_property_set_to_null_is_removed",
            reason: "without `temporal` a value set to null is stored as a null (#412 decision \
                     Q8: SET to null removes the property)",
        });
    }
    gaps
}

conformance_suite!(LpgStore, lpg_store, &gaps());
