---
title: Flutter Integration
description: Bundling Grafeo with Flutter desktop and mobile apps.
---

# Flutter Integration

Grafeo works as an embedded database inside Flutter apps via `dart:ffi`. This guide
covers building the native library and bundling it for each platform.

## Building grafeo-c

Build the shared library with the `lpg` profile, which includes GQL, Cypher,
Gremlin, SQL/PGQ, storage, and regex support:

```bash
cargo build --release -p grafeo-c --features lpg
```

Other available profiles include `embedded` (GQL + AI + algorithms), `rdf`
(SPARQL + GraphQL + triple store), and `full` (all languages + AI + triple
store). See the [feature flags documentation](https://grafeo.dev/reference/features/)
for details.

The output location depends on your host OS:

| OS      | Output path                             |
|---------|-----------------------------------------|
| Windows | `target/release/grafeo_c.dll`           |
| macOS   | `target/release/libgrafeo_c.dylib`      |
| Linux   | `target/release/libgrafeo_c.so`         |

## Platform Setup

### Windows

Copy `grafeo_c.dll` next to the Flutter runner executable:

```
my_app/
  windows/
    runner/
      grafeo_c.dll
```

To automate this in release builds, add to `windows/CMakeLists.txt` (before the
final `install` block):

```cmake
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/runner/grafeo_c.dll"
  DESTINATION "${INSTALL_BUNDLE_LIB_DIR}"
  COMPONENT Runtime)
```

### macOS

Copy `libgrafeo_c.dylib` into the macOS bundle:

```
my_app/
  macos/
    Frameworks/
      libgrafeo_c.dylib
```

Update the Xcode project (via `macos/Runner.xcodeproj`) to include the dylib in
the **Copy Bundle Resources** or **Embed Frameworks** build phase. The library
must be ad-hoc signed for local development:

```bash
codesign --force --sign - libgrafeo_c.dylib
```

### Linux

Copy `libgrafeo_c.so` into the bundle lib directory:

```
my_app/
  linux/
    bundle/
      lib/
        libgrafeo_c.so
```

The Flutter Linux runner sets `LD_LIBRARY_PATH` to include `bundle/lib/` at
startup, so the library is found automatically.

### Android / iOS

!!! warning "Mobile platforms are not yet supported"
    Prebuilt native binaries for iOS and Android are **not published**. The Dart
    FFI loader has platform detection for iOS (`DynamicLibrary.process()` for
    statically linked builds) and Android (`libgrafeo_c.so`), but you must
    cross-compile `grafeo-c` yourself using `cargo-ndk` (Android) or
    `cargo-lipo` (iOS). This is an advanced workflow that requires a working
    Rust cross-compilation toolchain for each target architecture.

Track progress in the [Grafeo roadmap](https://grafeo.dev/roadmap/).

## Custom Library Path

If the native library is not in the default search path, pass an explicit path:

```dart
final db = GrafeoDB.openSingleFile(
  dbPath,
  libraryPath: '/opt/grafeo/lib/libgrafeo_c.so',
);
```

All factory constructors (`memory`, `open`, `openSingleFile`, `openReadOnly`)
accept the optional `libraryPath` parameter.

## Recommended Pattern for Flutter Apps

Use `open` with the application documents directory from the
[`path_provider`](https://pub.dev/packages/path_provider) package
(`openSingleFile` is the same since 0.6):

```dart
import 'package:path_provider/path_provider.dart';
import 'package:grafeo/grafeo.dart';

Future<GrafeoDB> openDatabase() async {
  final dir = await getApplicationDocumentsDirectory();
  final dbPath = '${dir.path}/my_app.grafeo';
  return GrafeoDB.open(dbPath);
}
```

This stores all data in a single `.grafeo` file with a sidecar WAL for crash
safety.

## Isolate Considerations

All `GrafeoDB` FFI calls are synchronous and block the calling isolate. For
queries that may take more than a few milliseconds, run them in a separate
isolate using `Isolate.run` or `compute`:

```dart
final rows = await Isolate.run(() {
  final db = GrafeoDB.openReadOnly('/path/to/data.grafeo');
  try {
    final result = db.execute('MATCH (n:Person) RETURN n.name LIMIT 100');
    return result.rows;
  } finally {
    db.close();
  }
});
```

For write-heavy workloads, open the database once in a long-lived isolate and
communicate via `SendPort` / `ReceivePort`.
