/// Ordering tests for the fire-and-forget dispatch path.
///
/// The property under test is the one that makes `dispatchFire` usable at all:
/// binds join the same FIFO as dispatches, so bind → fire → rebind → fire
/// produces the same result as the fully awaited equivalent. A test that fires a
/// single dispatch against a static binding cannot see a violation — the rebind
/// between two fires is the whole point. (Before 1.5.9 the plain binds were
/// inline and this raced: every fired dispatch saw the LAST binding.)
@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

const _kAddOne = '''
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  data[gid.x] = data[gid.x] + 1u;
}
''';

/// Writes `src[i] + bias` into `dst[i]`. Buffers at explicit slots — the shape
/// tag-keyed binds cannot express, and the shape a texture-mixing shader needs.
// All bindings are read_write: minigpu's generated bind-group layout declares
// storage entries as read-write, and a `var<storage, read>` in the WGSL fails
// the layout match — which surfaces only as "[Invalid ComputePipeline] is
// invalid due to a previous error" on the next submit.
const _kAddBiasToDst = '''
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read_write> bias: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  dst[gid.x] = src[gid.x] + bias[0];
}
''';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  Future<Buffer> u32Buf(List<int> values) async {
    final buf = gpu.createBuffer(values.length * 4, BufferDataType.uint32);
    await buf.write(Uint32List.fromList(values), values.length,
        dataType: BufferDataType.uint32);
    return buf;
  }

  Future<List<int>> readU32(Buffer buf, int n) async {
    final out = Uint32List(n);
    await buf.read(out, n, dataType: BufferDataType.uint32);
    return out.toList();
  }

  test('dispatchFire is synchronized by a later awaited read', () async {
    final buf = await u32Buf([0, 0, 0, 0]);
    final shader = gpu.createComputeShader()..loadKernelString(_kAddOne);
    shader.setBufferAtSlot(0, buf);

    // Nothing is awaited between the fires; the read at the end is the only
    // synchronization point and must observe all of them.
    for (var i = 0; i < 8; i++) {
      shader.dispatchFire(4, 1, 1);
    }

    expect(await readU32(buf, 4), [8, 8, 8, 8]);

    shader.destroy();
    buf.destroy();
  });

  test('setBufferAtSlot rebind between fires is ordered', () async {
    // Each source is copied to its own destination by a rebind between fires.
    // If a rebind overtook a dispatch, a destination would show another
    // source's value; if it lagged, it would show a stale one.
    final srcA = await u32Buf([10, 10]);
    final srcB = await u32Buf([20, 20]);
    final srcC = await u32Buf([30, 30]);
    final dstA = await u32Buf([0, 0]);
    final dstB = await u32Buf([0, 0]);
    final dstC = await u32Buf([0, 0]);
    final bias = await u32Buf([1]);

    final shader = gpu.createComputeShader()..loadKernelString(_kAddBiasToDst);
    shader.setBufferAtSlot(2, bias);

    for (final (src, dst) in [(srcA, dstA), (srcB, dstB), (srcC, dstC)]) {
      shader
        ..setBufferAtSlot(0, src)
        ..setBufferAtSlot(1, dst)
        ..dispatchFire(2, 1, 1);
    }

    expect(await readU32(dstA, 2), [11, 11], reason: 'first bind pair');
    expect(await readU32(dstB, 2), [21, 21], reason: 'second bind pair');
    expect(await readU32(dstC, 2), [31, 31], reason: 'third bind pair');

    shader.destroy();
    for (final b in [srcA, srcB, srcC, dstA, dstB, dstC, bias]) {
      b.destroy();
    }
  });

  test('fired chain matches the fully awaited equivalent', () async {
    Future<List<int>> run({required bool fire}) async {
      final buf = await u32Buf([0, 1, 2, 3]);
      final shader = gpu.createComputeShader()..loadKernelString(_kAddOne);
      for (var i = 0; i < 5; i++) {
        shader.setBufferAtSlot(0, buf);
        if (fire) {
          shader.dispatchFire(4, 1, 1);
        } else {
          await shader.dispatch(4, 1, 1);
        }
      }
      final out = await readU32(buf, 4);
      shader.destroy();
      buf.destroy();
      return out;
    }

    expect(await run(fire: true), await run(fire: false));
  });

  test('dispatchFire enforces the 65535 workgroup cap', () {
    final shader = gpu.createComputeShader()..loadKernelString(_kAddOne);
    expect(
      () => shader.dispatchFire(ComputeShader.maxWorkgroupsPerDim + 1, 1, 1),
      throwsArgumentError,
    );
    shader.destroy();
  });
}
