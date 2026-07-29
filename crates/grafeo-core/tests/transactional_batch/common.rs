//! Shared fixtures for transactional batch storage tests.

#![allow(dead_code)]

use grafeo_common::types::{PropertyKey, TransactionId, Value};
use grafeo_core::graph::lpg::BatchNodeCreate;

/// A real (non-`SYSTEM`) transaction id so rows use `PENDING` visibility.
pub const TX: TransactionId = TransactionId::new(100);

pub fn node<'a>(labels: &'a [&'a str], props: Vec<(&'a str, Value)>) -> BatchNodeCreate<'a> {
    BatchNodeCreate {
        labels,
        properties: props
            .into_iter()
            .map(|(k, v)| (PropertyKey::new(k), v))
            .collect(),
    }
}
