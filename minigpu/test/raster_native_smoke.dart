// Public Dart/native raster path through the conditional minigpu_ffi adapter.
import 'dart:io';


import 'raster_scenario.dart';

Future<void> main() async {
  // Minigpu selects its native FFI adapter through dart.library.ffi.
  try {
    await runRasterScenario();
    stdout.writeln('raster native smoke: passed');
  } catch (error, stack) {
    stderr.writeln('raster native smoke: failed: $error');
    stderr.writeln(stack);
    exitCode = 1;
  }
}

