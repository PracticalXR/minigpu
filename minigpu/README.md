
# minigpu

A Flutter library for cross-platform GPU compute shaders integrating WGSL, GPU.CPP, and WebGPU via Dawn.

Try it: [https://minigpu.practicalxr.com/](https://minigpu.practicalxr.com/)

Use it: [https://pub.dev/packages/minigpu](https://pub.dev/packages/minigpu)

Tensor package: [https://pub.dev/packages/gpu_tensor](https://pub.dev/packages/gpu_tensor)

- [x] Windows
- [x] Linux
- [x] Mac
- [x] Web
- [x] Android
- [x] iOS

**Disclaimer: This library is experimental and supported at will, use in production at your own risk.**

## Three Things to Know

1. Dawn can take a while to build and our builds need a recent version of CMake. Run with -v to see errors and progress.
2. This package uses dart native assets.
For flutter, you must be on the master channel and run
`flutter config --enable-native-assets`
For dart, each run must contain the
`--enable-experiment=native-assets` flag.
3. For flutter web, add

```html
  <script src="assets/packages/minigpu_web/web/minigpu_web.loader.js"></script>
  <script>
    _minigpu.loader.load();
  </script>
```

to your web/index.html file.

### Installation

Make sure you have:

- Dart SDK >=3.9.0-21.0.dev or greater.
- Emscripten 4.0.10 or greater for web support.
- CMake 3.22 or greater.

Add the following to your `pubspec.yaml`:

```yaml
dependencies:
  minigpu: ^1.1.8
```

Then run:

```shell
dart pub get
```

## Getting Started

```console
 git clone https://github.com/PracticalXR/minigpu.git
 dart --enable-experiment=native-assets test

 dart:
 cd minigpu
 dart --enable-experiment=native-assets bin/example.dart

 flutter:
 cd minigpu/example
 flutter config --enable-native-assets
 flutter run -d chrome/windows/linux/android/ios
```

## Example

 ```dart

Future<void> _runGELU() async {
  // Initialize the GPU.
  final gpu = Minigpu();
  await gpu.init();
  // Create the compute shader.
  final shader = gpu.createComputeShader();

// Load the compute kernel code as a string.
  shader.loadKernelString('''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
) {
  let i: u32 = GlobalInvocationID.x;
  if (i < arrayLength(&inp)) {
    let x: f32 = inp[i];
    out[i] = select(
      0.5 * x * (1.0 + tanh(
        GELU_SCALING_FACTOR * (x + .044715 * x * x * x)
      )),
      x,
      x > 10.0
    );
  }
}
''');

// Define the buffer size and generate input data.
  final bufferSize = 100;
  final inputData = Float32List.fromList(
    List<double>.generate(bufferSize, (i) => i / 10.0),
  );
  print('bufferSize: ${inputData.lengthInBytes}');
  final memSize = bufferSize * 4; // 4 bytes per float32

// Create GPU buffers for input and output data.
  final inputBuffer = gpu.createBuffer(bufferSize, memSize);
  final outputBuffer = gpu.createBuffer(bufferSize, memSize);

// Upload the input data.
 await inputBuffer.write(inputData, bufferSize);

// Bind the buffers to the shader.
  shader.setBuffer('inp', inputBuffer);
  shader.setBuffer('out', outputBuffer);

// Calculate the number of workgroups required.
  final workgroups = ((bufferSize + 255) / 256).floor();

// Dispatch the compute shader.
  await shader.dispatch(workgroups, 1, 1);

// Read the output data.
  final outputData = Float32List(bufferSize);
  await outputBuffer.read(outputData, bufferSize);

// Update the UI 
  setState(() {
    final result = outputData.sublist(0, 16).map((value) => value.toDouble()).toList();
    print('Result: $result');
  });
}
  ```

## Funding

  If you are interested in funding further easy-to-port gpu development, please submit an inquiry on [https://practicalxr.com](https://practicalxr.com).
