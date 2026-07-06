# GPU Tensor

GPU Tensor is a Dart package for performing tensor operations leveraging GPU acceleration. It offers efficient slicing, element access, arithmetic operations, and FFT computations.

- [x] Windows
- [x] Linux
- [x] Mac - Needs testing.
- [x] Web
- [x] Android
- [x] iOS - Needs testing.

**Disclaimer: This library is experimental and supported at will, use in production at your own risk.**

## Features

- **GPU-Accelerated:** Implemented using the minigpu package which compiles and uses webgpu for shader execution.
- **Basic Operations:** +, -, *, /, and %.
- **Scalar Operations:** Scalar  +, -, *, /, and %.
- **Linear Operations:** Matrix multiplication and convolution.
- **Data Operations** Slice, reshape, getElement, setElement, head, and tail.
- **Transforms:** .fft() up to 3D.
- **Activation Functions:** Relu, Sigmoid,  Sin, Cos, Tanh, and Softmax.
- **Pooling Operations**: Basic implementations of Max and Min pooling.
- **Formatted Prints** for tensors with head and tail helpers.


**Missing something important?** Open an issue please, PRs are welcome too! You can also create an extension on Tensor in your own code.

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

Add the following to your `pubspec.yaml`:

```yaml
dependencies:
  gpu_tensor: ^1.0.0
```

Then run:

```console
dart pub get
```

## Example

Below is a quick example that demonstrates creating tensors, performing arithmetic operations, slicing, and using the formatted `head`/`tail` helpers:

```dart
// Import the package
import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';

Future<void> main() async {
  // Create a 3x3 tensor (Tensor A) with values 1..9.
  final a = await Tensor.create(
    [3, 3],
    data: Float32List.fromList([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
    ]),
  );

  // Create another 3x3 tensor (Tensor B) with values 9..1.
  final b = await Tensor.create(
    [3, 3],
    data: Float32List.fromList([
      9, 8, 7,
      6, 5, 4,
      3, 2, 1,
    ]),
  );

  // Elementwise addition:
  final added = await (a + b);
  final addedData = await added.getData();
  print("Elementwise addition result: $addedData");

  // Elementwise subtraction:
  final subtracted = await (a - b);
  final subtractedData = await subtracted.getData();
  print("Elementwise subtraction result: $subtractedData");

  // Scalar multiplication:
  final scalarMult = await a.multiplyScalar(2.0);
  final scalarMultData = await scalarMult.getData();
  print("Scalar multiplication result: $scalarMultData");

  // Matrix multiplication:
  final matMulResult = await a.matMul(b);
  final matMulData = await matMulResult.getData();
  print("Matrix multiplication result: $matMulData");

  // Reshape the multiplication result into a 1-D tensor:
  final reshaped = matMulResult.reshape([9]);
  final reshapedData = await reshaped.getData();
  print("Reshaped matrix multiplication result: $reshapedData");

  // Get an individual element:
  // For a 3x3 matrix, element at indices [1,2] should be 6.
  final elementA = await a.getElement([1, 2]);
  print("Element at [1,2] in Tensor A: $elementA");

  // Use the head() helper to show the first 2 rows and 2 columns.
  final headA = await a.head([2, 2]);
  print("Head of Tensor A (first 2 rows, 2 cols):\n$headA");

  // Use the tail() helper to show the last 2 rows and 2 columns.
  final tailA = await a.tail([2, 2]);
  print("Tail of Tensor A (last 2 rows, 2 cols):\n$tailA");

  // FFT demo (1D FFT)
  const int points1D = 8;
  final Float32List realData1D = Float32List(points1D);
  for (int i = 0; i < points1D; i++) {
    realData1D[i] = i.toDouble();
  }
  final realTensor1D = await Tensor.create([points1D], data: realData1D);
  final fft1dResult = await realTensor1D.fft1d();
  final fft1dResultData = await fft1dResult.getData();
  print("FFT 1D result: $fft1dResultData");

  // FFT demo (2D FFT)
  const int rows = 4;
  const int cols = 4;
  final Float32List realData2D = Float32List(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    realData2D[i] = i.toDouble();
  }
  final realTensor2D = await Tensor.create([rows, cols], data: realData2D);
  final fft2dResult = await realTensor2D.fft2d();
  final fft2dResultData = await fft2dResult.getData();
  print("FFT 2D result: $fft2dResultData");
}
```

## Performance tips

### Reuse readback buffers

`getData()` allocates a new `TypedData` on every call. In a tight render or encode loop, supply a pre-allocated buffer to avoid per-frame GC pressure:

```dart
// Allocate once
final scratch = Float32List(tensor.size);

// Reuse every frame — no allocation, no GC pressure
final data = await tensor.getData(into: scratch) as Float32List;
```

The `into` buffer must have at least `tensor.size × elementSizeBytes` bytes; an `ArgumentError` is thrown if it is too small.

## Funding

  If you are interested in funding further easy-to-port gpu development, please submit an inquiry on [https://practicalXR.com](https://practicalxr.com).
