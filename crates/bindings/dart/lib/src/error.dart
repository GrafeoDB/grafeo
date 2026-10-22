/// Error handling for the Grafeo Dart binding.
///
/// Status codes match the C enum in `grafeo-c/src/error.rs` exactly.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'ffi/bindings.dart';

/// Status codes returned by grafeo-c FFI functions.
///
/// Values must match the C `GrafeoStatus` enum:
///   Ok=0, ErrorDatabase=1, ErrorQuery=2, ErrorTransaction=3, ErrorStorage=4,
///   ErrorIo=5, ErrorSerialization=6, ErrorInternal=7, ErrorNullPointer=8,
///   ErrorInvalidUtf8=9.
enum GrafeoStatus {
  ok(0),
  database(1),
  query(2),
  transaction(3),
  storage(4),
  io(5),
  serialization(6),
  internal(7),
  nullPointer(8),
  invalidUtf8(9);

  final int code;
  const GrafeoStatus(this.code);

  static GrafeoStatus fromCode(int code) =>
      values.firstWhere((s) => s.code == code, orElse: () => internal);
}

/// Base exception for all Grafeo errors.
sealed class GrafeoException implements Exception {
  final String message;
  final GrafeoStatus status;

  /// The GQLSTATUS of the error: the five-character ISO GQL status of the
  /// condition, such as `22012` (division by zero) or `42001` (syntax error).
  /// Null for an exception the binding raises itself.
  final String? gqlStatus;
  const GrafeoException(this.message, this.status, {this.gqlStatus});

  @override
  String toString() => '$runtimeType(${status.name}): $message';
}

/// A query parsing or execution error (status 2).
class QueryException extends GrafeoException {
  const QueryException(super.message, super.status, {super.gqlStatus});
}

/// A transaction error such as conflict or invalid state (status 3).
class TransactionException extends GrafeoException {
  const TransactionException(super.message, super.status, {super.gqlStatus});
}

/// A storage or IO error (status 4, 5).
class StorageException extends GrafeoException {
  const StorageException(super.message, super.status, {super.gqlStatus});
}

/// A serialization error (status 6).
class SerializationException extends GrafeoException {
  const SerializationException(super.message, super.status, {super.gqlStatus});
}

/// A generic database error (status 1, 7, 8, 9, or unknown). A write to a
/// closed database (GRAFEO-T007 in the message) and a call after a commit that
/// did not complete (GRAFEO-T008, reopen the database) raise this one, not a
/// [TransactionException]: no retry fixes them.
class DatabaseException extends GrafeoException {
  /// Creates a [DatabaseException] with [message] and [status].
  const DatabaseException(super.message, super.status, {super.gqlStatus});
}

/// Map a C status code and error message to a typed Dart exception.
///
/// Mirrors `grafeo-bindings-common::error::classify_error`.
GrafeoException classifyError(int statusCode, String message,
    {String? gqlStatus}) {
  final status = GrafeoStatus.fromCode(statusCode);
  return switch (status) {
    GrafeoStatus.query => QueryException(message, status, gqlStatus: gqlStatus),
    GrafeoStatus.transaction =>
      TransactionException(message, status, gqlStatus: gqlStatus),
    GrafeoStatus.storage ||
    GrafeoStatus.io =>
      StorageException(message, status, gqlStatus: gqlStatus),
    GrafeoStatus.serialization =>
      SerializationException(message, status, gqlStatus: gqlStatus),
    _ => DatabaseException(message, status, gqlStatus: gqlStatus),
  };
}

/// Read the last error message from the grafeo-c thread-local error slot.
///
/// The returned pointer is owned by the C library (thread-local storage) and
/// is valid until the next FFI call on this thread. We copy it to a Dart
/// string immediately and do NOT free it.
String lastError(GrafeoBindings bindings) {
  final ptr = bindings.grafeoLastError();
  if (ptr == nullptr) return 'Unknown error';
  return ptr.toDartString();
}

/// Read the GQLSTATUS of the last error from the grafeo-c thread-local slot,
/// or null when there is none.
String? lastGqlStatus(GrafeoBindings bindings) {
  final ptr = bindings.grafeoLastErrorGqlStatus();
  if (ptr == nullptr) return null;
  return ptr.toDartString();
}

/// Throw a [GrafeoException] for a failed FFI call that returned a status code.
Never throwStatus(GrafeoBindings bindings, int statusCode) {
  throw classifyError(statusCode, lastError(bindings),
      gqlStatus: lastGqlStatus(bindings));
}

/// Throw a [GrafeoException] for a failed FFI call that returned null.
Never throwLastError(GrafeoBindings bindings) {
  throw DatabaseException(
    lastError(bindings),
    GrafeoStatus.database,
    gqlStatus: lastGqlStatus(bindings),
  );
}
