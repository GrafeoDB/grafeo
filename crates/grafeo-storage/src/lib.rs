//! # grafeo-storage
//!
//! Persistence layer for Grafeo: WAL, container format, and section I/O.
//!
//! This crate handles all disk I/O for Grafeo. It depends only on
//! `grafeo-common` (never on `grafeo-core` or `grafeo-adapters`),
//! treating section data as opaque bytes.
//!
//! ## Modules
//!
//! - [`wal`] - Write-ahead log for durability
//! - [`lock`] - Exclusive lock for WAL-directory databases
//! - [`mod@file`] - Single-file `.grafeo` format with crash-safe dual headers

#![deny(unsafe_code)]

pub mod container;

#[cfg(feature = "wal")]
pub mod wal;

#[cfg(feature = "wal")]
pub mod lock;

#[cfg(feature = "grafeo-file")]
pub mod file;

#[cfg(feature = "async-storage")]
pub mod async_backend;

#[cfg(feature = "async-storage")]
pub mod async_local;
