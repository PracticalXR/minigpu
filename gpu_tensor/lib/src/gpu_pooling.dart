import 'package:gpu_tensor/src/gpu_helpers.dart';

import 'gpu_tensor_base.dart';

/// Shared implementation for max/min pooling — the two differ only in the
/// accumulator init/update/output strings.
///
/// NOTE: pooling kernels operate on f32 data (no dtype templating yet).
Future<Tensor> _pool(
  Tensor tensor, {
  required List<int> poolSizes,
  List<int>? strides,
  List<int>? pads,
  List<int>? poolAxes,
  required String accumulatorInit,
  required String accumulatorUpdate,
  required String outputTransform,
}) async {
  final shape = tensor.shape;
  int effectiveRank = shape.length;
  int numPool = poolSizes.length;
  strides ??= List.filled(numPool, 1);
  pads ??= List.filled(numPool, 0);
  poolAxes ??= (effectiveRank == 3)
      ? [0, 1]
      : List.generate(numPool, (i) => effectiveRank - numPool + i);
  if (poolAxes.length != numPool) {
    throw Exception("poolAxes length must equal poolSizes length");
  }
  for (int ax in poolAxes) {
    if (ax < 0 || ax >= effectiveRank) {
      throw Exception("poolAxes values must be between 0 and effectiveRank-1");
    }
  }

  // Build output shape by replacing dimensions corresponding to pooling axes.
  List<int> outputShape = List.from(shape);
  for (int j = 0; j < numPool; j++) {
    int ax = poolAxes[j];
    int outDim = ((shape[ax] + pads[j] - poolSizes[j]) ~/ strides[j]) + 1;
    outputShape[ax] = outDim;
  }
  Tensor result = await Tensor.create(outputShape, gpu: tensor.gpu);

  // Compute input strides (row-major).
  List<int> inStrides = List.filled(effectiveRank, 1);
  for (int i = effectiveRank - 2; i >= 0; i--) {
    inStrides[i] = inStrides[i + 1] * shape[i + 1];
  }
  // Compute output strides.
  List<int> outStrides = List.filled(effectiveRank, 1);
  for (int i = effectiveRank - 2; i >= 0; i--) {
    outStrides[i] = outStrides[i + 1] * outputShape[i + 1];
  }
  int totalOut = outputShape.reduce((a, b) => a * b);

  final constants = StringBuffer();
  for (int i = 0; i < effectiveRank; i++) {
    constants.writeln("const in_$i: u32 = ${shape[i]}u;");
  }
  for (int j = 0; j < numPool; j++) {
    constants.writeln("const stride_$j: u32 = ${strides[j]}u;");
    constants.writeln("const pad_$j: u32 = ${pads[j]}u;");
    constants.writeln("const poolSize_$j: u32 = ${poolSizes[j]}u;");
  }

  final coordDecode = StringBuffer();
  for (int i = 0; i < effectiveRank; i++) {
    coordDecode.writeln("  let coord_$i: u32 = idx_rem / ${outStrides[i]}u;");
    coordDecode.writeln("  idx_rem = idx_rem % ${outStrides[i]}u;");
  }

  final loopOpen = StringBuffer();
  for (int j = 0; j < numPool; j++) {
    loopOpen.writeln(
        "  for (var p_$j: u32 = 0u; p_$j < poolSize_$j; p_$j = p_$j + 1u) {");
  }
  final loopClose = StringBuffer();
  for (int j = 0; j < numPool; j++) {
    loopClose.writeln("  }");
  }

  final indexTerms = <String>[];
  for (int i = 0; i < effectiveRank; i++) {
    int pAxis = poolAxes.indexOf(i);
    if (pAxis == -1) {
      indexTerms.add("(i32(coord_$i) * i32(${inStrides[i]}))");
    } else {
      indexTerms.add(
          "((i32(coord_$i) * i32(${strides[pAxis]})) - i32(${pads[pAxis]}) + i32(p_$pAxis)) * i32(${inStrides[i]})");
    }
  }

  final maskChecks = StringBuffer();
  for (int i = 0; i < effectiveRank; i++) {
    int pAxis = poolAxes.indexOf(i);
    if (pAxis != -1) {
      maskChecks.writeln(
          "    let in_${i}_coord: i32 = (i32(coord_$i) * i32(stride_$pAxis)) - i32(pad_$pAxis) + i32(p_$pAxis);");
      maskChecks.writeln(
          "    if (in_${i}_coord < 0 || u32(in_${i}_coord) >= in_$i) { localMask = 0.0; }");
    }
  }

  // The input load sits INSIDE the mask guard: padding sub-samples used to
  // perform a genuinely out-of-range read (WebGPU clamps, but it's wasted
  // bandwidth and relies on clamping semantics).
  String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const totalOut: u32 = ${totalOut}u;
$constants

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (idx >= totalOut) {
    return;
  }

  var idx_rem: u32 = idx;
$coordDecode

  $accumulatorInit

$loopOpen
    let inIndexTemp: i32 = ${indexTerms.join(" + ")};

    var localMask: f32 = 1.0;
$maskChecks

    if (localMask > 0.0) {
      let val: f32 = input[u32(inIndexTemp)];
      $accumulatorUpdate
    }
$loopClose

  $outputTransform
}
''';

  final shader = tensor.gpu.cachedShader(shaderCode);
  shader.setBuffer('input', tensor.buffer);
  shader.setBuffer('output', result.buffer);
  await shader.dispatchLinear(totalOut);
  return result;
}

extension TensorPoolingMax on Tensor {
  Future<Tensor> maxPool({
    required List<int> poolSizes,
    List<int>? strides,
    List<int>? pads,
    List<int>? poolAxes,
  }) {
    return _pool(
      this,
      poolSizes: poolSizes,
      strides: strides,
      pads: pads,
      poolAxes: poolAxes,
      accumulatorInit: "var maxVal: f32 = -3.4e38;",
      accumulatorUpdate: "if(val > maxVal) { maxVal = val; }",
      outputTransform: "output[idx] = maxVal;",
    );
  }
}

extension TensorPoolingMin on Tensor {
  Future<Tensor> minPool({
    required List<int> poolSizes,
    List<int>? strides,
    List<int>? pads,
    List<int>? poolAxes,
  }) {
    return _pool(
      this,
      poolSizes: poolSizes,
      strides: strides,
      pads: pads,
      poolAxes: poolAxes,
      accumulatorInit: "var minVal: f32 = 3.4e38;",
      accumulatorUpdate: "if(val < minVal) { minVal = val; }",
      outputTransform: "output[idx] = minVal;",
    );
  }
}
