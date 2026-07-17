import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';

import 'gpu_nn.dart';
import 'gpu_quant.dart';

/// A Mixture-of-Experts FFN layer (qwen35moe structure):
///
///   probs   = softmax(router @ x)             // router: [experts, dim] f32
///   top-k   = highest-k probs (renormalized when [normTopK])
///   expert  = down_e( silu(gate_e(x)) * up_e(x) )   // quantized stacks
///   out     = sum_k w_k * expert_k
///           + sigmoid(sharedGate . x) * down_s(silu(gate_s(x)) * up_s(x))
///
/// Routing (top-k over `experts` floats) happens on the CPU — it is a
/// readback of one small vector per layer; the expert math stays on GPU via
/// expert-indexed fused matVec.
class MoeFfn {
  MoeFfn({
    required this.router,
    required this.gateExps,
    required this.upExps,
    required this.downExps,
    required this.topK,
    this.normTopK = true,
    this.gateShexp,
    this.upShexp,
    this.downShexp,
    this.sharedGate,
  }) {
    if (gateExps.experts != upExps.experts ||
        gateExps.experts != downExps.experts) {
      throw Exception("Expert stack counts disagree");
    }
    if (topK < 1 || topK > gateExps.experts) {
      throw Exception("topK $topK out of range (1..${gateExps.experts})");
    }
  }

  /// Router weight, shape [experts, dim] float32.
  final Tensor router;

  /// Expert stacks: gate/up are [experts, ff, dim], down is [experts, dim, ff].
  final QuantizedTensor gateExps;
  final QuantizedTensor upExps;
  final QuantizedTensor downExps;

  final int topK;

  /// Renormalize the selected top-k probabilities to sum to 1 (Qwen-style).
  final bool normTopK;

  /// Optional shared expert (same shapes as one expert) + its sigmoid gate
  /// vector [dim].
  final QuantizedTensor? gateShexp;
  final QuantizedTensor? upShexp;
  final QuantizedTensor? downShexp;
  final Tensor? sharedGate;

  int get experts => gateExps.experts;
  int get dim => gateExps.cols;
  int get ff => gateExps.rows;

  /// Routing decision for [x]: indices + weights of the selected experts.
  /// Exposed separately so tests (and later, the expert-cache prefetcher)
  /// can observe routing.
  Future<({List<int> indices, List<double> weights})> route(Tensor x) async {
    final xCol = x.reshape([dim, 1]);
    final logits = await router.matMul(xCol); // [experts, 1]
    final probsT = await logits.reshape([1, experts]).softmax();
    final probs = await probsT.getData() as Float32List;
    logits.destroy();
    probsT.destroy();

    final order = List<int>.generate(experts, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));
    final indices = order.take(topK).toList();
    var weights = indices.map((i) => probs[i].toDouble()).toList();
    if (normTopK) {
      final sum = weights.fold(0.0, (a, b) => a + b);
      if (sum > 0) {
        weights = weights.map((w) => w / sum).toList();
      }
    }
    return (indices: indices, weights: weights);
  }

  /// One SiLU-gated expert FFN: down( silu(gate(x)) * up(x) ).
  Future<Tensor> _expertFfn(
    Tensor x,
    QuantizedTensor gate,
    QuantizedTensor up,
    QuantizedTensor down,
    int expert,
  ) async {
    final g = await gate.matVec(x, expert: expert);
    final gAct = await g.silu();
    g.destroy();
    final u = await up.matVec(x, expert: expert);
    final prod = await gAct.multiply(u);
    gAct.destroy();
    u.destroy();
    final out = await down.matVec(prod, expert: expert);
    prod.destroy();
    return out;
  }

  /// Forward pass for a single-token activation [x] of shape [dim].
  /// Returns a fresh [dim] tensor.
  Future<Tensor> forward(Tensor x) async {
    if (x.size != dim) {
      throw Exception("MoeFfn.forward: x has ${x.size} elements, dim is $dim");
    }
    final routing = await route(x);

    Tensor? acc;
    for (int k = 0; k < routing.indices.length; k++) {
      final e = routing.indices[k];
      final w = routing.weights[k];
      final out = await _expertFfn(x, gateExps, upExps, downExps, e);
      final scaled = await out.multiplyScalar(w);
      out.destroy();
      if (acc == null) {
        acc = scaled;
      } else {
        final next = await acc.add(scaled);
        acc.destroy();
        scaled.destroy();
        acc = next;
      }
    }

    if (gateShexp != null) {
      // Scalar sigmoid gate: sigmoid(sharedGate . x).
      final gCol = sharedGate!.reshape([1, dim]);
      final xCol = x.reshape([dim, 1]);
      final dotT = await gCol.matMul(xCol); // [1, 1]
      final dot = (await dotT.getData() as Float32List)[0];
      dotT.destroy();
      final gateVal = 1.0 / (1.0 + math.exp(-dot));

      final sh = await _expertFfn(x, gateShexp!, upShexp!, downShexp!, 0);
      final shScaled = await sh.multiplyScalar(gateVal);
      sh.destroy();
      final next = await acc!.add(shScaled);
      acc.destroy();
      shScaled.destroy();
      acc = next;
    }

    return acc!;
  }

  void destroy() {
    gateExps.destroy();
    upExps.destroy();
    downExps.destroy();
    gateShexp?.destroy();
    upShexp?.destroy();
    downShexp?.destroy();
    router.destroy();
    sharedGate?.destroy();
  }
}
