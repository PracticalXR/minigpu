import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';

Future<void> main() async {
  // Initialize the library.
  final minigpu = Minigpu();
  await minigpu.init();

  // Buffer configuration.
  const int bufferSize = 100;
  final int floatSize = 4; // 4 bytes per f32
  final int memorySize = bufferSize * floatSize;

  // Create input and output buffers.
  final inputBuffer = minigpu.createBuffer(memorySize, BufferDataType.float32);
  final outputBuffer = minigpu.createBuffer(memorySize, BufferDataType.float32);

  // Initialize input data.
  List<double> data = List.generate(bufferSize, (i) => i / 10.0);
  final Float32List inputData = Float32List.fromList(data);
  await inputBuffer.write(inputData, bufferSize);

  // WGSL shader code from the Flutter example.
  final shaderCode = '''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        //out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + .044715 * x * x * x))), x, x > 10.0);
        out[i] = inp[i] + .2;
    }
}
''';

  // Create and configure the compute shader.
  final shader = minigpu.createComputeShader();
  shader.loadKernelString(shaderCode);
  shader.setBuffer('inp', inputBuffer);
  shader.setBuffer('out', outputBuffer);

  // Calculate workgroup count (assuming 256 threads per workgroup).
  int workgroups = ((bufferSize + 255) / 256).floor();
  await shader.dispatch(workgroups, 1, 1);

  // Read and print the output data.
  final Float32List outputData = Float32List(bufferSize);
  await outputBuffer.read(outputData, bufferSize);
  print('Output data: $outputData');

  // Clean up resources.
  shader.destroy();
  inputBuffer.destroy();
  outputBuffer.destroy();
}
