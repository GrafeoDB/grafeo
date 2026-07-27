//! Shared fixtures for session transactional batch tests.

#![allow(dead_code)]

use grafeo_common::types::Value;
use grafeo_engine::session::TransactionalNodeCreate;

pub fn person(name: &str, age: i64) -> TransactionalNodeCreate {
    TransactionalNodeCreate::new(["Person"])
        .with_property("name", Value::from(name))
        .with_property("age", Value::from(age))
}
