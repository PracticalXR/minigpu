import 'dart:async';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:test/test.dart';

void main() {
  // Only printStats() prints; starting, stopping and disposing are silent.
  test('the monitor does not print on its own', () {
    final printed = <String>[];
    runZoned(
      () {
        final monitor = MainThreadMonitor();
        monitor.startMonitoring();
        monitor.startMonitoring();
        monitor.stopMonitoring();
        monitor.stopMonitoring();
        MainThreadMonitor.disposeSingleton();
      },
      zoneSpecification: ZoneSpecification(
        print: (self, parent, zone, line) => printed.add(line),
      ),
    );
    expect(printed, isEmpty);
    expect(MainThreadMonitor().isMonitoring, isFalse);
  });

  // The monitor is process-wide and runs a 5 ms timer while on. Pipelines
  // used to start it on construction and never stop it, so the timer outlived
  // every pipeline and kept the isolate alive.
  test('a pipeline leaves the event-loop monitor off', () async {
    final pipeline = Pipeline(id: 'monitor_off');
    expect(pipeline.threadMonitor.isMonitoring, isFalse);
    await pipeline.dispose();
    expect(MainThreadMonitor().isMonitoring, isFalse);
  });
}
