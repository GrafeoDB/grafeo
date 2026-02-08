//! C FFI bindings for the Grafeo graph database.
//!
//! This crate exposes a C-compatible API that can be consumed by any language
//! with C interop support (Go via CGO, Ruby via FFI, Java via JNI, etc.).
//!
//! # Memory Management
//!
//! All opaque pointers returned by `grafeo_*` functions must be freed by their
//! corresponding `grafeo_free_*` function. Strings returned by the API are owned
//! by the Rust side and must be freed with [`grafeo_free_string`].
//!
//! # Error Handling
//!
//! Functions return [`GrafeoStatus`] codes. On error, call [`grafeo_last_error`]
//! to retrieve a human-readable error message. The error is thread-local and
//! valid until the next FFI call on the same thread.
//!
//! # Thread Safety
//!
//! A [`GrafeoDatabase`] handle can be shared across threads. Internally it uses
//! `Arc<RwLock<GrafeoDB>>` for safe concurrent access.

#![allow(unsafe_code)]

mod database;
mod error;
mod types;
