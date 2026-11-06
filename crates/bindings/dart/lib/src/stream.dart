/// Lazy, cursor-based GQL result streaming.
///
/// [ResultStream] wraps the grafeo-c streaming FFI. Pulling a row calls
/// `grafeo_stream_next_row_json` and decodes the returned JSON into a
/// `Map<String, dynamic>`. The stream finishes when the C side reports
/// exhaustion (status Ok with null row); exceptions from the engine are
/// raised as [DatabaseException].
///
/// Call [close] when done (or rely on [NativeFinalizer] as a safety net) to
/// release the underlying Rust iterator.
///
/// New in 0.5.40. Experimental: signature may change before Beta.
library;

import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'ffi/bindings.dart';
import 'types.dart';
import 'value.dart';

/// Async cursor over the rows of a read-only GQL query.
///
/// Rows are delivered one at a time as `Map<String, dynamic>`. Memory stays
/// bounded regardless of result-set size.
///
/// Rejects mutating queries (INSERT / DELETE / SET), EXPLAIN / PROFILE,
/// session/schema commands, and queries that compile to push-based pipelines
/// (ORDER BY, aggregate, DISTINCT). Use [GrafeoDB.execute] for those.
class ResultStream implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  final List<String> _columns;
  bool _closed = false;

  /// What the stream had when it was closed.
  List<QueryWarning> _warnings = const [];

  static NativeFinalizer? _finalizer;

  ResultStream._(this._handle, this._columns, this._bindings) {
    _finalizer ??= NativeFinalizer(
      _bindings.library.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'grafeo_stream_free',
      ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  /// Opens a streaming query. Not a public constructor: callers should go
  /// through `GrafeoDB.executeStream`.
  static ResultStream open(
    Pointer<Void> dbHandle,
    GrafeoBindings bindings,
    String query,
  ) {
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoStreamOpen(dbHandle, queryPtr);
      if (ptr == nullptr) throwLastError(bindings);
      // _readColumns can throw; the NativeFinalizer is only attached by the
      // ResultStream constructor, so on failure we must free the handle here.
      final List<String> columns;
      try {
        columns = _readColumns(ptr, bindings);
      } catch (_) {
        bindings.grafeoStreamFree(ptr);
        rethrow;
      }
      return ResultStream._(ptr, columns, bindings);
    } finally {
      malloc.free(queryPtr);
    }
  }

  static List<String> _readColumns(
    Pointer<Void> handle,
    GrafeoBindings bindings,
  ) {
    final strPtr = bindings.grafeoStreamColumnsJson(handle);
    if (strPtr == nullptr) throwLastError(bindings);
    try {
      final decoded = jsonDecode(strPtr.toDartString());
      if (decoded is! List) return const <String>[];
      return decoded.whereType<String>().toList(growable: false);
    } finally {
      bindings.grafeoFreeString(strPtr);
    }
  }

  /// Column names in the order they appear in each row map.
  List<String> get columns => _columns;

  /// Pulls the next row, or returns `null` when the stream is exhausted.
  ///
  /// Throws [DatabaseException] if the engine reports an error.
  Map<String, dynamic>? next() {
    _checkOpen();
    final out = malloc<Pointer<Utf8>>();
    out.value = nullptr;
    try {
      final status = _bindings.grafeoStreamNextRowJson(_handle, out);
      if (status != 0) {
        throwLastError(_bindings);
      }
      final rowPtr = out.value;
      if (rowPtr == nullptr) {
        return null; // exhausted
      }
      try {
        final decoded = jsonDecode(rowPtr.toDartString());
        if (decoded is Map) {
          // Each value decoded as a query row's is ($bytes, $decimal, ...)
          return decodeObject(decoded);
        }
        throw DatabaseException(
          'Streaming row is not a JSON object: ${rowPtr.toDartString()}',
          GrafeoStatus.serialization,
        );
      } finally {
        _bindings.grafeoFreeString(rowPtr);
      }
    } finally {
      malloc.free(out);
    }
  }

  /// Iterates the remaining rows via a synchronous generator. Convenient for
  /// `for (final row in stream.rows())` loops.
  Iterable<Map<String, dynamic>> rows() sync* {
    while (true) {
      final row = next();
      if (row == null) return;
      yield row;
    }
  }

  /// Drains the stream into a list. Convenient for small result sets or
  /// tests; beats the purpose of streaming for large ones.
  List<Map<String, dynamic>> toList() {
    final out = <Map<String, dynamic>>[];
    for (final row in rows()) {
      out.add(row);
    }
    return out;
  }

  /// The warnings the statement raised so far, ordered by code: its
  /// translation's from the open, its run's as the rows that raise them are
  /// pulled, all of them once the stream is exhausted. After [close] it
  /// holds what the stream had then. Experimental (0.6.2+).
  List<QueryWarning> get warnings => _closed ? _warnings : _readWarnings();

  List<QueryWarning> _readWarnings() => readWarnings(
        _bindings.grafeoStreamWarningCount(_handle),
        (index) => _bindings.grafeoStreamWarningCode(_handle, index),
        (index) => _bindings.grafeoStreamWarningKind(_handle, index),
        (index) => _bindings.grafeoStreamWarningMessage(_handle, index),
      );

  /// Explicitly releases the underlying Rust iterator.
  void close() {
    if (_closed) return;
    _warnings = _readWarnings();
    _closed = true;
    _finalizer!.detach(this);
    _bindings.grafeoStreamFree(_handle);
    _handle = nullptr;
  }

  void _checkOpen() {
    if (_closed) {
      throw DatabaseException(
        'ResultStream is closed',
        GrafeoStatus.database,
      );
    }
  }
}
