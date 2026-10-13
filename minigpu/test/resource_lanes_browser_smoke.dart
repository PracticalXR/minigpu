// Public Dart/Wasm resource lanes against the generated WebGPU backend. The
// result is left in `bufferSmokeResult`, and each lane's record in
// `resourceLanesResult`, for the page's runner to read.
//
// `?back-to-back=<rounds>` in the page address adds the stage that destroys a
// buffer and allocates its replacement at once.
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'resource_lanes_scenario.dart';

Future<void> main() async {
  var result = 'passed';
  final record = <String, Object>{};
  try {
    final rounds =
        int.tryParse(Uri.base.queryParameters['back-to-back'] ?? '') ?? 0;
    record['buffer_lane'] = await runBufferLaneScenario(
      replacementRounds: rounds,
    );
    record['texture_lane'] = await runTextureLaneScenario();
    record['owned_lifetime'] = await runOwnedLifetimeScenario();
    // A page has one context: the foreign-context case does not exist there.
    record['foreign_context'] = await runForeignContextScenario()
        ? 'refused'
        : 'not applicable';
  } catch (error, stack) {
    result = 'failed: $error\n$stack';
  }
  globalContext.setProperty(
    'resourceLanesResult'.toJS,
    jsonEncode(record).toJS,
  );
  globalContext.setProperty('bufferSmokeResult'.toJS, result.toJS);
}
