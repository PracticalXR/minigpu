// Public Dart/native resource lanes through the conditional minigpu_ffi
// adapter. Prints one line per lane and a final verdict.
//
// `back-to-back[=rounds]` as an argument adds the stage that destroys a buffer
// and allocates its replacement at once (200 rounds unless a count is given).
// Leave it out on the Dawn D3D12 backend unless the native library allocates
// buffers on its worker: the scenario's own comment says why.
import 'dart:convert';
import 'dart:io';

import 'resource_lanes_scenario.dart';

Future<void> main(List<String> arguments) async {
  var rounds = 0;
  for (final argument in arguments) {
    if (argument == 'back-to-back') {
      rounds = 200;
    } else if (argument.startsWith('back-to-back=')) {
      rounds = int.parse(argument.substring('back-to-back='.length));
    } else {
      stderr.writeln('resource lanes native smoke: unknown argument $argument');
      exitCode = 2;
      return;
    }
  }
  try {
    final buffers = await runBufferLaneScenario(replacementRounds: rounds);
    stdout.writeln('buffer lane: ${jsonEncode(buffers)}');
    final textures = await runTextureLaneScenario();
    stdout.writeln('texture lane: ${jsonEncode(textures)}');
    final lifetime = await runOwnedLifetimeScenario();
    stdout.writeln('owned lifetime: ${jsonEncode(lifetime)}');
    if (!await runForeignContextScenario()) {
      throw StateError('No second context was available on this adapter');
    }
    stdout.writeln('foreign context: refused');
    stdout.writeln('resource lanes native smoke: passed');
  } catch (error, stack) {
    stderr.writeln('resource lanes native smoke: failed: $error');
    stderr.writeln(stack);
    exitCode = 1;
  }
}
