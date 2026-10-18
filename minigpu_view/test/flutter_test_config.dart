import 'dart:async';

// Flutter's windowing flag is internal while the feature is experimental.
// ignore: implementation_imports
import 'package:flutter/src/foundation/_features.dart';

/// The machine's Flutter config enables windowing, which the single-view test
/// binding cannot start under flutter_tester.
Future<void> testExecutable(FutureOr<void> Function() testMain) async {
  // ignore: invalid_use_of_internal_member
  isWindowingEnabled = false;
  await testMain();
}
