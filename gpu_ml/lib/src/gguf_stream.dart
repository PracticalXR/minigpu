/// Streaming GGUF access for large local model files (dart:io).
/// See gpu_ml_io.dart for the public entrypoint.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

import '../gpu_ml.dart';
/// A GGUF file opened for streaming access.
///
/// [header] exposes the parsed metadata and tensor directory (same
/// [GgufFile] API as the in-memory path, minus `tensorBytes`, which would
/// require whole-file bytes — use [readTensorBytes] instead).
class GgufStream {
  GgufStream._(this.path, this.header, this._raf, this._fileLength);

  final String path;
  final GgufFile header;
  final RandomAccessFile _raf;
  final int _fileLength;

  int get fileLength => _fileLength;

  /// Opens [path] and parses the header.  [maxHeaderBytes] bounds the
  /// header+metadata+directory read (tokenizer vocab metadata can reach tens
  /// of MB on large-vocab models).
  static Future<GgufStream> open(
    String path, {
    int maxHeaderBytes = 512 * 1024 * 1024,
  }) async {
    final file = File(path);
    final len = await file.length();
    final headerLen = len < maxHeaderBytes ? len : maxHeaderBytes;
    final raf = await file.open();
    final headBytes = Uint8List(headerLen);
    await raf.setPosition(0);
    final read = await raf.readInto(headBytes);
    if (read < headerLen) {
      await raf.close();
      throw Exception("Short read on '$path': $read of $headerLen bytes");
    }
    final GgufFile header;
    try {
      header = GgufFile.parse(headBytes);
    } catch (_) {
      await raf.close();
      rethrow;
    }
    return GgufStream._(path, header, raf, len);
  }

  GgufTensorInfo? tensor(String name) => header.tensor(name);
  List<GgufTensorInfo> get tensors => header.tensors;
  Map<String, dynamic> get metadata => header.metadata;

  /// Reads [info]'s raw (packed) bytes from disk.  Only the requested range
  /// is read.
  ///
  /// [byteOffset]/[byteLength] select a sub-range WITHIN the tensor's data —
  /// the streaming primitive for loading one expert of a MoE stack or the
  /// first N rows of a huge matrix (blocks are row-contiguous, so any
  /// whole-row/whole-expert range is a valid packed sub-tensor).
  Future<Uint8List> readTensorBytes(
    GgufTensorInfo info, {
    int byteOffset = 0,
    int? byteLength,
  }) async {
    final total = info.byteSize;
    final size = byteLength ?? (total - byteOffset);
    if (byteOffset < 0 || size < 0 || byteOffset + size > total) {
      throw Exception(
        "Range [$byteOffset, ${byteOffset + size}) outside tensor '${info.name}' ($total bytes)",
      );
    }
    final start = header.dataOffset + info.offset + byteOffset;
    if (start + size > _fileLength) {
      throw Exception(
        "Tensor '${info.name}' data [$start, ${start + size}) exceeds file size $_fileLength",
      );
    }
    final out = Uint8List(size);
    await _raf.setPosition(start);
    int done = 0;
    while (done < size) {
      final n = await _raf.readInto(
        Uint8List.sublistView(out, done, size),
      );
      if (n <= 0) {
        throw Exception(
          "Short read for tensor '${info.name}': $done of $size bytes",
        );
      }
      done += n;
    }
    return out;
  }

  /// Loads a quantized/f16 weight (2D, or 3D expert stack) by [name]
  /// straight from disk to VRAM.
  Future<QuantizedTensor> loadQuantized(String name, {Minigpu? gpu}) async {
    final info = tensor(name);
    if (info == null) {
      throw Exception("GGUF tensor '$name' not found");
    }
    if (info.ne.length != 2 && info.ne.length != 3) {
      throw Exception(
        "loadQuantized supports 2D/3D tensors; '$name' has ne ${info.ne}",
      );
    }
    final bytes = await readTensorBytes(info);
    return QuantizedTensor.create(info.shape, info.type, bytes, gpu: gpu);
  }

  /// Loads an f32 tensor by [name] as a regular [Tensor].
  Future<Tensor> loadF32(String name, {Minigpu? gpu}) async {
    final info = tensor(name);
    if (info == null) {
      throw Exception("GGUF tensor '$name' not found");
    }
    if (info.type != GgmlType.f32) {
      throw Exception(
        "Tensor '$name' has ggml type ${info.type}, not f32 — use loadQuantized",
      );
    }
    final bytes = await readTensorBytes(info);
    // Copy for alignment: a view over arbitrary offsets may not be 4-aligned.
    final data = Float32List.sublistView(Uint8List.fromList(bytes));
    return Tensor.create(info.shape, data: data, gpu: gpu);
  }

  Future<void> close() => _raf.close();

  /// Loads block [blk]'s full-attention layer (qwen35moe naming/metadata).
  Future<AttentionLayer> loadAttentionLayer(int blk, {Minigpu? gpu}) async {
    final arch = metadata['general.architecture'];
    final heads = metadata['$arch.attention.head_count'] as int;
    final kvHeads = metadata['$arch.attention.head_count_kv'] as int;
    final headDim = metadata['$arch.attention.key_length'] as int;
    final ropeDims = metadata['$arch.rope.dimension_count'] as int;
    final thetaBase =
        (metadata['$arch.rope.freq_base'] as num?)?.toDouble() ?? 10000.0;
    final eps = (metadata['$arch.attention.layer_norm_rms_epsilon'] as num?)
            ?.toDouble() ??
        1e-6;

    return AttentionLayer(
      wq: await loadQuantized('blk.$blk.attn_q.weight', gpu: gpu),
      wk: await loadQuantized('blk.$blk.attn_k.weight', gpu: gpu),
      wv: await loadQuantized('blk.$blk.attn_v.weight', gpu: gpu),
      wo: await loadQuantized('blk.$blk.attn_output.weight', gpu: gpu),
      qNorm: await loadF32('blk.$blk.attn_q_norm.weight', gpu: gpu),
      kNorm: await loadF32('blk.$blk.attn_k_norm.weight', gpu: gpu),
      heads: heads,
      kvHeads: kvHeads,
      headDim: headDim,
      ropeDims: ropeDims,
      ropeThetaBase: thetaBase,
      eps: eps,
    );
  }

  /// Loads block [blk]'s Gated DeltaNet layer (qwen35moe naming/metadata),
  /// allocating fresh zero conv/recurrent state.
  Future<DeltaNetLayer> loadDeltaNetLayer(int blk, {Minigpu? gpu}) async {
    final arch = metadata['general.architecture'];
    final headDim = metadata['$arch.ssm.state_size'] as int;
    final kHeads = metadata['$arch.ssm.group_count'] as int;
    final vHeads = metadata['$arch.ssm.time_step_rank'] as int;
    final convKernel = metadata['$arch.ssm.conv_kernel'] as int;
    final eps = (metadata['$arch.attention.layer_norm_rms_epsilon'] as num?)
            ?.toDouble() ??
        1e-6;

    final ssmAT = await loadF32('blk.$blk.ssm_a', gpu: gpu);
    final dtBiasT = await loadF32('blk.$blk.ssm_dt.bias', gpu: gpu);
    final ssmA = await ssmAT.getData() as Float32List;
    final dtBias = await dtBiasT.getData() as Float32List;
    ssmAT.destroy();
    dtBiasT.destroy();

    final convDim = 2 * kHeads * headDim + vHeads * headDim;
    return DeltaNetLayer(
      wqkv: await loadQuantized('blk.$blk.attn_qkv.weight', gpu: gpu),
      wGate: await loadQuantized('blk.$blk.attn_gate.weight', gpu: gpu),
      wBeta: await loadQuantized('blk.$blk.ssm_beta.weight', gpu: gpu),
      wAlpha: await loadQuantized('blk.$blk.ssm_alpha.weight', gpu: gpu),
      convWeight: await loadF32('blk.$blk.ssm_conv1d.weight', gpu: gpu),
      ssmA: Float32List.fromList(ssmA),
      dtBias: Float32List.fromList(dtBias),
      ssmNorm: await loadF32('blk.$blk.ssm_norm.weight', gpu: gpu),
      wOut: await loadQuantized('blk.$blk.ssm_out.weight', gpu: gpu),
      kHeads: kHeads,
      vHeads: vHeads,
      headDim: headDim,
      convKernel: convKernel,
      convState: await Tensor.create([convKernel - 1, convDim], gpu: gpu),
      ssmState: await Tensor.create([vHeads * headDim * headDim], gpu: gpu),
      eps: eps,
    );
  }

  /// Loads block [blk]'s complete MoE FFN (qwen35moe tensor naming) from
  /// disk to VRAM: router + the three expert stacks + shared expert + its
  /// gate.  [topK] comes from `<arch>.expert_used_count` metadata when not
  /// given.
  Future<MoeFfn> loadMoeFfn(int blk, {int? topK, Minigpu? gpu}) async {
    final arch = metadata['general.architecture'];
    topK ??= (metadata['$arch.expert_used_count'] as int?) ?? 8;

    Future<QuantizedTensor> q(String suffix) =>
        loadQuantized('blk.$blk.$suffix', gpu: gpu);

    final router = await loadF32('blk.$blk.ffn_gate_inp.weight', gpu: gpu);
    final sharedGateInfo = tensor('blk.$blk.ffn_gate_inp_shexp.weight');
    final hasShared = sharedGateInfo != null;

    return MoeFfn(
      router: router,
      gateExps: await q('ffn_gate_exps.weight'),
      upExps: await q('ffn_up_exps.weight'),
      downExps: await q('ffn_down_exps.weight'),
      topK: topK,
      gateShexp: hasShared ? await q('ffn_gate_shexp.weight') : null,
      upShexp: hasShared ? await q('ffn_up_shexp.weight') : null,
      downShexp: hasShared ? await q('ffn_down_shexp.weight') : null,
      sharedGate: hasShared
          ? await loadF32('blk.$blk.ffn_gate_inp_shexp.weight', gpu: gpu)
          : null,
    );
  }
}
