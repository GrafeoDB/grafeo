/// ACID transaction support for Grafeo.
///
/// Uses [NativeFinalizer] to auto-rollback (via Rust Drop) if neither
/// [commit] nor [rollback] is called.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'ffi/bindings.dart';
import 'native.dart';
import 'types.dart';
import 'value.dart';

/// An ACID transaction on a Grafeo database.
///
/// Obtain via [GrafeoDB.beginTransaction]. Must be explicitly committed
/// or rolled back. If dropped without either, the Rust Drop impl
/// performs an automatic rollback.
class Transaction implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  bool _finished = false;

  /// A commit was tried and failed: it ended the transaction without its
  /// changes, so a rollback has nothing to do.
  bool _commitFailed = false;

  static NativeFinalizer? _finalizer;

  /// Create a transaction wrapper around a native handle.
  ///
  /// Typically called by [GrafeoDB.beginTransaction], not directly.
  Transaction(this._handle, this._bindings) {
    _finalizer ??= NativeFinalizer(
      _bindings.library.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'grafeo_free_transaction',
      ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  void _checkActive() {
    if (_finished) {
      throw TransactionException(
        'Transaction already finished',
        GrafeoStatus.transaction,
      );
    }
  }

  /// Execute a GQL query within this transaction.
  QueryResult execute(String query) {
    _checkActive();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.grafeoTransactionExecute(_handle, queryPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
    }
  }

  /// Execute a parameterized GQL query within this transaction.
  QueryResult executeWithParams(String query, Map<String, dynamic> params) {
    _checkActive();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    final paramsJson = encodeParams(params);
    final paramsPtr = paramsJson.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.grafeoTransactionExecuteWithParams(
        _handle,
        queryPtr,
        paramsPtr,
      );
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
      malloc.free(paramsPtr);
    }
  }

  /// Execute a query in the given [language] within this transaction.
  ///
  /// [language] is one of: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
  /// `"sparql"`, `"sql"`. Omit [params] for queries without parameters.
  QueryResult executeLanguage(
    String language,
    String query, {
    Map<String, dynamic>? params,
  }) {
    _checkActive();
    final langPtr = language.toNativeUtf8(allocator: malloc);
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    Pointer<Utf8>? paramsPtr;
    if (params != null) {
      paramsPtr = encodeParams(params).toNativeUtf8(allocator: malloc);
    }
    try {
      final resultPtr = _bindings.grafeoTransactionExecuteLanguage(
        _handle,
        langPtr,
        queryPtr,
        paramsPtr ?? nullptr.cast<Utf8>(),
      );
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(langPtr);
      malloc.free(queryPtr);
      if (paramsPtr != null) malloc.free(paramsPtr);
    }
  }

  /// Commit the transaction, making all changes permanent. A commit that
  /// fails (a write conflict, GQLSTATUS 40000) ends the transaction all the
  /// same.
  void commit() {
    _checkActive();
    final status = _bindings.grafeoCommit(_handle);
    if (status != GrafeoStatus.ok.code) {
      final error = _endUnlessClosed(status);
      if (_finished) _commitFailed = true;
      throw error;
    }
    _end();
  }

  /// Ends the transaction after a failed commit or rollback, unless the
  /// database was closed: the native layer then answers every later call
  /// with GRAFEO-T007, so the transaction stays for it. Returns the error.
  GrafeoException _endUnlessClosed(int status) {
    final error = classifyError(
      status,
      lastError(_bindings),
      gqlStatus: lastGqlStatus(_bindings),
    );
    if (!error.message.startsWith('GRAFEO-T007')) _end();
    return error;
  }

  /// Marks the transaction finished and frees its native transaction.
  void _end() {
    _finished = true;
    _finalizer!.detach(this);
    _bindings.grafeoFreeTransaction(_handle);
    _handle = nullptr;
  }

  /// Rollback the transaction, discarding all changes. After a [commit]
  /// that failed it does nothing, as in the other bindings: that commit
  /// ended the transaction, so a rollback in a catch block does not hide
  /// the commit's error. After the database closed it throws GRAFEO-T007,
  /// as every call does.
  void rollback() {
    if (_commitFailed) return;
    _checkActive();
    final status = _bindings.grafeoRollback(_handle);
    if (status != GrafeoStatus.ok.code) throw _endUnlessClosed(status);
    _end();
  }

  QueryResult _buildResult(Pointer<Void> resultPtr) =>
      buildResult(_bindings, resultPtr);
}
