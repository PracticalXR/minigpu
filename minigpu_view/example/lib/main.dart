/// minigpu_view example — GPU zero-copy preview
///
/// Demonstrates rendering a [SharedOutputTexture] produced by a minigpu
/// compute pass directly into a Flutter widget with no CPU readback.
///
/// The example runs a simple WGSL compute shader that writes a
/// time-animated gradient into a [SharedOutputTexture] at ~60 fps and
/// displays the result using [MiniavGpuPreview].
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:minigpu/minigpu.dart';
import 'package:minigpu_view/minigpu_view.dart';

void main() {
  runApp(const MinigpuViewExampleApp());
}

class MinigpuViewExampleApp extends StatelessWidget {
  const MinigpuViewExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'minigpu_view example',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const GpuPreviewPage(),
    );
  }
}

// ---------------------------------------------------------------------------
// WGSL compute shader — writes an animated f32 RGBA gradient into a storage
// buffer.  Each pixel is 4 floats: (r, g, b, 1.0).
// ---------------------------------------------------------------------------
const String _kGradientShader = '''
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read_write> params: array<f32>; // [W, H, time, 0]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = u32(params[0]);
  let H = u32(params[1]);
  if (gid.x >= W || gid.y >= H) { return; }

  let uv   = vec2<f32>(f32(gid.x) / f32(W), f32(gid.y) / f32(H));
  let t    = params[2];
  let base = (gid.y * W + gid.x) * 4u;

  out[base + 0u] = 0.5 + 0.5 * sin(uv.x * 6.2831 + t);
  out[base + 1u] = 0.5 + 0.5 * sin(uv.y * 6.2831 + t + 2.094);
  out[base + 2u] = 0.5 + 0.5 * sin((uv.x + uv.y) * 4.0 + t + 4.189);
  out[base + 3u] = 1.0;
}
''';

// ---------------------------------------------------------------------------

class GpuPreviewPage extends StatefulWidget {
  const GpuPreviewPage({super.key});

  @override
  State<GpuPreviewPage> createState() => _GpuPreviewPageState();
}

class _GpuPreviewPageState extends State<GpuPreviewPage> {
  static const int _kWidth = 640;
  static const int _kHeight = 360;

  late final Minigpu _gpu;
  late final ComputeShader _shader;
  late final Buffer _outputBuffer;
  late final Buffer _paramBuffer;
  late final SharedOutputTexture _sharedTex;
  late final MinigpuPreviewController _controller;
  Timer? _timer;
  double _time = 0.0;
  bool _ready = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _controller = MinigpuPreviewController();
    _init();
  }

  Future<void> _init() async {
    try {
      _gpu = Minigpu();
      await _gpu.init();

      // Output buffer: _kWidth * _kHeight * 4 f32 values (RGBA).
      final int outByteSize = _kWidth * _kHeight * 4 * 4;
      _outputBuffer = _gpu.createBuffer(outByteSize, BufferDataType.float32);

      // Param buffer: vec4 [W, H, time, 0].
      final int paramByteSize = 4 * 4;
      _paramBuffer = _gpu.createBuffer(paramByteSize, BufferDataType.float32);
      await _paramBuffer.write(
        Float32List.fromList([
          _kWidth.toDouble(),
          _kHeight.toDouble(),
          0.0,
          0.0,
        ]),
        4,
      );

      _shader = _gpu.createComputeShader();
      _shader.loadKernelString(_kGradientShader);
      _shader.setBuffer('out', _outputBuffer);
      _shader.setBuffer('params', _paramBuffer);

      // Zero-copy shared texture for display in Flutter.
      final tex = _gpu.createSharedOutputTexture(_kWidth, _kHeight);
      if (tex == null)
        throw Exception('SharedOutputTexture not supported on this platform');
      _sharedTex = tex;

      setState(() => _ready = true);

      // Render loop — ~60 fps.
      _timer = Timer.periodic(const Duration(milliseconds: 16), (_) => _tick());
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  Future<void> _tick() async {
    if (!mounted) return;
    _time += 0.016;

    await _paramBuffer.write(
      Float32List.fromList([
        _kWidth.toDouble(),
        _kHeight.toDouble(),
        _time,
        0.0,
      ]),
      4,
    );

    // Re-bind buffers each frame (tag map is already set, dispatch uses it).
    await _shader.dispatch((_kWidth + 7) ~/ 8, (_kHeight + 7) ~/ 8, 1);

    // GPU blit: float-buffer → shared D3D11/Metal texture, no CPU copy.
    _sharedTex.copyFromBufferF32(_outputBuffer);

    // Hand the texture handle to Flutter via the platform channel.
    await _controller.present(_sharedTex.asPreviewSource());
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    if (_ready) {
      _sharedTex.destroy();
      _outputBuffer.destroy();
      _paramBuffer.destroy();
      _shader.destroy();
      _gpu.destroy();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(
              'GPU init failed:\n$_error',
              style: const TextStyle(color: Colors.red),
            ),
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('minigpu_view — GPU zero-copy preview')),
      body: Column(
        children: [
          // GPU preview — fills available width, preserves aspect ratio.
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: AspectRatio(
                aspectRatio: _kWidth / _kHeight,
                child: MiniavGpuPreview(
                  controller: _controller,
                  fit: BoxFit.fill,
                  placeholder: const Center(child: CircularProgressIndicator()),
                ),
              ),
            ),
          ),

          // Status row.
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: ListenableBuilder(
              listenable: _controller,
              builder: (context, _) {
                final id = _controller.textureId;
                final sz = _controller.size;
                return Text(
                  id == null
                      ? 'Waiting for first frame…'
                      : 'texture #$id  •  ${sz.width.toInt()}×${sz.height.toInt()}  '
                            '•  t=${_time.toStringAsFixed(2)}',
                  style: Theme.of(context).textTheme.bodySmall,
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
