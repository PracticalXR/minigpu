// Web stub for [PersistentIsolateExecutor] — selected by the conditional
// export in `stage_executor.dart` on platforms without `dart:io`.  The web
// has no isolate spawning; stages there must use [InlineExecutor] (the
// default).  Construction succeeds so shared code can be written once;
// actually RUNNING work throws with a clear message.
import 'dart:typed_data';

import 'stage_executor.dart' show StageExecutor, StageProcessor;

class PersistentIsolateExecutor extends StageExecutor {
  PersistentIsolateExecutor(this.processor);

  final StageProcessor processor;

  @override
  String get backend => 'isolate-unsupported';

  @override
  Future<Map<String, TypedData>> run(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> parameters,
  ) {
    throw UnsupportedError(
      'PersistentIsolateExecutor is not available on the web — '
      'use InlineExecutor (isolateEnabled: false) instead.',
    );
  }
}
