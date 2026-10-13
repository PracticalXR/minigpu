// Public Dart/native raster lifecycle checks through the conditional
// minigpu_ffi adapter. Prints one line per check and a final verdict.
import 'dart:convert';
import 'dart:io';

import 'raster_lifecycle_scenario.dart';

Future<void> main() async {
  try {
    final cache = await runRasterBindingCacheScenario();
    stdout.writeln('binding cache: ${jsonEncode(cache)}');
    final closed = await runRasterPendingCloseScenario();
    stdout.writeln('pending close: ${jsonEncode(closed)}');
    if (!await runRasterForeignOwnerScenario()) {
      throw StateError('No second context was available on this adapter');
    }
    stdout.writeln('foreign owner: refused');
    // The preparation-timeout check needs a runner that can withhold the
    // device's pipeline completions. No native backend offers one.
    stdout.writeln('preparation timeout: not run (nothing can withhold it)');
    stdout.writeln('raster lifecycle native smoke: passed');
  } catch (error, stack) {
    stderr.writeln('raster lifecycle native smoke: failed: $error');
    stderr.writeln(stack);
    exitCode = 1;
  }
}
