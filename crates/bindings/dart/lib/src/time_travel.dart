/// Time travel: reads of the database as it was at an earlier epoch or time.
library;

import 'dart:ffi';

import 'error.dart';
import 'ffi/bindings.dart';
import 'native.dart';
import 'types.dart';
import 'value.dart';

/// A point in the database's history: an epoch ([TimePoint.epoch]), a time
/// ([TimePoint.at]) or a duration before now ([TimePoint.ago]). Times have
/// the resolution of a millisecond.
final class TimePoint {
  // The GRAFEO_POINT_* kinds of the C API.
  static const _epochKind = 0;
  static const _timeKind = 1;
  static const _agoKind = 2;

  /// The C API kind of the point.
  final int kind;

  /// The C API value of the point: an epoch or milliseconds.
  final int value;

  const TimePoint._(this.kind, this.value);

  /// The committed state of [epoch]; a later epoch reads now.
  factory TimePoint.epoch(int epoch) {
    RangeError.checkNotNegative(epoch, 'epoch');
    return TimePoint._(_epochKind, epoch);
  }

  /// The last commit at or before [time]. A local and a UTC [DateTime] both
  /// name one instant.
  factory TimePoint.at(DateTime time) =>
      TimePoint._(_timeKind, time.millisecondsSinceEpoch);

  /// The last commit at or before [duration] ago.
  factory TimePoint.ago(Duration duration) {
    RangeError.checkNotNegative(duration.inMilliseconds, 'duration');
    return TimePoint._(_agoKind, duration.inMilliseconds);
  }
}

/// How much history a database keeps for time-travel reads: none
/// ([HistoryRetention.off], the default), all of it
/// ([HistoryRetention.forever]) or what was committed within a duration
/// before now ([HistoryRetention.duration]). A database file stores the
/// setting; one given at open replaces it.
final class HistoryRetention {
  // The GRAFEO_HISTORY_* kinds of the C API.
  static const _offKind = 0;
  static const _foreverKind = 1;
  static const _durationKind = 2;

  /// The C API kind of the retention.
  final int kind;

  /// The retained duration in milliseconds, for [HistoryRetention.duration].
  final int durationMs;

  const HistoryRetention._(this.kind, this.durationMs);

  /// No history: only the current epoch can be read.
  static const off = HistoryRetention._(_offKind, 0);

  /// All history.
  static const forever = HistoryRetention._(_foreverKind, 0);

  /// The history committed within [duration] before now.
  factory HistoryRetention.duration(Duration duration) {
    RangeError.checkNotNegative(duration.inMilliseconds, 'duration');
    return HistoryRetention._(_durationKind, duration.inMilliseconds);
  }
}

/// A read-only view of the database as it was at an earlier epoch, from
/// [GrafeoDB.asOf]. Its queries see the committed state of [epoch]; a query
/// that writes throws. It keeps the history of its epoch until [close]d, with
/// the [NativeFinalizer] as a safety net.
class AsOfView implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  bool _closed = false;

  /// The epoch the view reads at.
  final int epoch;

  static NativeFinalizer? _finalizer;

  /// Wraps a native view handle; typically called by [GrafeoDB.asOf].
  AsOfView(this._handle, this._bindings)
      : epoch = _bindings.grafeoAsOfEpoch(_handle) {
    _finalizer ??= NativeFinalizer(
      _bindings.library.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
        'grafeo_free_as_of',
      ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  /// Release the view and the history it keeps. Safe to call multiple times.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer!.detach(this);
    _bindings.grafeoFreeAsOf(_handle);
    _handle = nullptr;
  }

  Pointer<Void> get _open {
    if (_closed) {
      throw DatabaseException('AsOfView is closed', GrafeoStatus.database);
    }
    return _handle;
  }

  /// Execute a GQL query, with optional [params], as the database was at the
  /// view's epoch.
  QueryResult execute(String query, {Map<String, dynamic>? params}) =>
      withNativeStrings([query, params == null ? null : encodeParams(params)],
          (p) {
        final resultPtr = _bindings.grafeoAsOfExecute(_open, p[0], p[1]);
        if (resultPtr == nullptr) throwLastError(_bindings);
        return buildResult(_bindings, resultPtr);
      });
}
