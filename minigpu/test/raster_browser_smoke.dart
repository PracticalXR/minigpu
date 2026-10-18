// Public Dart/Wasm raster path against the generated WebGPU backend.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'raster_scenario.dart';

Future<void> main() async {
  var result = 'passed';
  try {
    await runRasterScenario();
  } catch (error, stack) {
    result = 'failed: $error\n$stack';
  }
  globalContext.setProperty('bufferSmokeResult'.toJS, result.toJS);
}

