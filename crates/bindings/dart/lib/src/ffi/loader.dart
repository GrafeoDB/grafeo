/// Platform-specific native library loading for grafeo-c.
library;

import 'dart:ffi';
import 'dart:io';

/// Cached native library instance, loaded once per process.
final DynamicLibrary nativeLibrary = _load(null);

/// Load the native library, optionally from a specific path.
///
/// When [path] is null the loader tries the platform-specific library name
/// directly, which works when the shared library is on the system library path,
/// next to the executable, or bundled by Flutter.
DynamicLibrary loadNativeLibrary([String? path]) {
  if (path != null) {
    return DynamicLibrary.open(path);
  }
  return nativeLibrary;
}

DynamicLibrary _load(String? path) {
  if (path != null) {
    return DynamicLibrary.open(path);
  }

  if (Platform.isIOS) {
    // On iOS the native code is statically linked into the runner.
    return DynamicLibrary.process();
  }

  final String libraryName;
  if (Platform.isWindows) {
    libraryName = 'grafeo_c.dll';
  } else if (Platform.isMacOS) {
    libraryName = 'libgrafeo_c.dylib';
  } else if (Platform.isLinux || Platform.isAndroid) {
    libraryName = 'libgrafeo_c.so';
  } else {
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  return DynamicLibrary.open(libraryName);
}
