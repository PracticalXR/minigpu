import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:collection';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'serialization/pipeline_models.dart';

/// Types of assets that can be loaded
enum AssetType {
  json,
  text,
  binary,
  weights,
  texture,
  tensor,
  lookupTable,
  shader,
  unknown,
}

/// Source types for assets
enum AssetSourceType { file, url, embedded, generated, preloaded }

/// Asset loading strategies (not strictly enforced by tests)
enum AssetLoadingStrategy { preload, lazy, perCall, stream }

/// Caching policies for assets
enum AssetCachePolicy { keep, evictAfterUse }

/// Events emitted by asset manager
class AssetEvent {
  final String kind; // 'loaded' | 'failed' | 'released' | 'evicted'
  final String name;
  final String? message;
  AssetEvent(this.kind, this.name, {this.message});
}

/// Concrete StaticAsset
class StaticAsset {
  final String name;
  final AssetType type;

  // Payload
  Uint8List? bytes;
  String? text;
  Tensor? tensor;

  // Optional source info (for stage-provided-file factories)
  final Map<String, dynamic>? source;

  // Optional metadata
  final Map<String, dynamic>? metadata;

  // Requiredness hint
  final bool isRequired;

  StaticAsset({
    required this.name,
    required this.type,
    this.bytes,
    this.text,
    this.tensor,
    this.source,
    this.metadata,
    this.isRequired = true,
  });

  // Common factories used by StageBuilder
  factory StaticAsset.weights(
    String name, {
    required String path,
    bool isRequired = true,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: AssetType.weights,
    source: {'type': 'file', 'path': path},
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.texture({
    required String name,
    required String path,
    bool isRequired = true,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: AssetType.texture,
    source: {'type': 'file', 'path': path},
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.config({
    required String name,
    required String path,
    bool isRequired = true,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: AssetType.json,
    source: {'type': 'file', 'path': path},
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.lookupTable({
    required String name,
    required String path,
    bool isRequired = true,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: AssetType.lookupTable,
    source: {'type': 'file', 'path': path},
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.generated({
    required String name,
    required AssetType type,
    Map<String, dynamic> generatorParams = const {},
    bool isRequired = true,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: type,
    source: {'type': 'generated', 'params': generatorParams},
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.bytes({
    required String name,
    required AssetType type,
    required Uint8List data,
    bool isRequired = false,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: type,
    bytes: data,
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.text({
    required String name,
    required String data,
    bool isRequired = false,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: AssetType.text,
    text: data,
    metadata: metadata,
    isRequired: isRequired,
  );

  factory StaticAsset.tensor({
    required String name,
    required Tensor data,
    bool isRequired = false,
    Map<String, dynamic>? metadata,
  }) => StaticAsset(
    name: name,
    type: AssetType.tensor,
    tensor: data,
    metadata: metadata,
    isRequired: isRequired,
  );
}

class _Entry {
  StaticAsset asset;
  int refCount;
  final bool persistent;
  _Entry({required this.asset, this.refCount = 1, this.persistent = true});
}

AssetSourceType _parseSourceType(String? s) {
  switch (s) {
    case 'file':
      return AssetSourceType.file;
    case 'url':
      return AssetSourceType.url;
    case 'embedded':
      return AssetSourceType.embedded;
    case 'generated':
      return AssetSourceType.generated;
    case 'preloaded':
      return AssetSourceType.preloaded;
    default:
      return AssetSourceType.embedded;
  }
}

AssetType _inferType(String? hint, Map<String, dynamic> source) {
  if (hint != null) {
    switch (hint.toLowerCase()) {
      case 'json':
        return AssetType.json;
      case 'text':
        return AssetType.text;
      case 'binary':
      case 'bin':
        return AssetType.binary;
      case 'weights':
        return AssetType.weights;
      case 'texture':
        return AssetType.texture;
      case 'tensor':
        return AssetType.tensor;
      case 'lut':
      case 'lookuptable':
        return AssetType.lookupTable;
      case 'shader':
      case 'wgsl':
      case 'glsl':
        return AssetType.text; // treat as text for tests
      default:
        break;
    }
  }

  // Infer from file extension
  final path = source['path'] as String?;
  if (path != null) {
    final p = path.toLowerCase();
    if (p.endsWith('.json')) return AssetType.json;
    if (p.endsWith('.txt')) return AssetType.text;
    if (p.endsWith('.wgsl') || p.endsWith('.glsl')) return AssetType.text;
    if (p.endsWith('.bin') || p.endsWith('.dat')) return AssetType.binary;
    if (p.endsWith('.png') ||
        p.endsWith('.jpg') ||
        p.endsWith('.jpeg') ||
        p.endsWith('.bmp')) {
      return AssetType.texture;
    }
  }

  // Embedded heuristic: default to binary unless clearly text
  final data = source['data'];
  if (data is String) {
    // If it's valid JSON, mark as json, else text
    try {
      json.decode(data);
      return AssetType.json;
    } catch (_) {
      return AssetType.text;
    }
  }

  return AssetType.unknown;
}

// Add simple detectors for JSON and base64-like strings.
bool _looksLikeJsonString(String s) {
  final t = s.trimLeft();
  return t.startsWith('{') || t.startsWith('[');
}

bool _looksLikeBase64(String s) {
  final t = s.trim();
  if (t.isEmpty || t.length % 4 != 0) return false;
  final base64Regex = RegExp(r'^[A-Za-z0-9+/]+={0,2}$');
  return base64Regex.hasMatch(t);
}

Uint8List _asBytes(dynamic data) {
  if (data == null) return Uint8List(0);
  if (data is Uint8List) return data;
  if (data is List<int>) return Uint8List.fromList(data);
  if (data is String) {
    // Try base64 first, else treat as utf8
    try {
      return base64.decode(data);
    } catch (_) {
      return Uint8List.fromList(utf8.encode(data));
    }
  }
  throw ArgumentError('Unsupported embedded data type: ${data.runtimeType}');
}

class DisposableToken {
  final Future<void> Function() _dispose;
  DisposableToken(this._dispose);
  Future<void> dispose() => _dispose();
}

class AssetManager {
  final LinkedHashMap<String, _Entry> _cache = LinkedHashMap<String, _Entry>();
  final StreamController<AssetEvent> _events =
      StreamController<AssetEvent>.broadcast(sync: true);

  Stream<AssetEvent> get events => _events.stream;

  // Cache sizing
  int? _maxBytes;
  int _currentBytes = 0;

  // Helper: emit without crashing if the stream is closed.
  void _safeEmit(String kind, String name, {String? message}) {
    try {
      _events.add(AssetEvent(kind, name, message: message));
    } catch (_) {
      // ignore if closed
    }
  }

  Future<void> setCacheLimit(int maxBytes) async {
    if (maxBytes <= 0) {
      _maxBytes = null; // disable limit
      return;
    }
    _maxBytes = maxBytes;
    _evictIfNeeded();
  }

  // Compute approx size contribution of an asset.
  int _assetSize(StaticAsset a) {
    if (a.bytes != null) return a.bytes!.lengthInBytes;
    if (a.text != null) return utf8.encode(a.text!).length;
    // Tensor size is unknown without dtype/shape; treat as 0 for now.
    return 0;
  }

  void _touch(String name) {
    final e = _cache.remove(name);
    if (e != null) _cache[name] = e;
  }

  void _evictIfNeeded() {
    if (_maxBytes == null) return;
    if (_currentBytes <= _maxBytes!) return;
    // Evict from oldest to newest, only non-persistent with refCount == 0.
    final toRemove = <String>[];
    for (final entry in _cache.entries) {
      if (_currentBytes <= _maxBytes!) break;
      final name = entry.key;
      final e = entry.value;
      if (!e.persistent && e.refCount == 0) {
        _currentBytes -= _assetSize(e.asset);
        toRemove.add(name);
      }
    }
    for (final n in toRemove) {
      _cache.remove(n);
      _safeEmit('evicted', n);
    }
  }

  void clearAsset(String name) {
    final removed = _cache.remove(name);
    if (removed != null) {
      _currentBytes -= _assetSize(removed.asset);
      _safeEmit('evicted', name);
    }
  }

  void clearCache() {
    final names = _cache.keys.toList();
    _cache.clear();
    _currentBytes = 0;
    for (final n in names) {
      _safeEmit('evicted', n);
    }
  }

  Future<void> dispose() async {
    // Clear first (may emit), then close stream.
    clearCache();
    await _events.close();
  }

  // Stage-provided single asset preload
  Future<void> preload(StaticAsset asset) async {
    final mat = await _materializeIfNeeded(asset);
    final existing = _cache[mat.name];
    if (existing != null) {
      _currentBytes -= _assetSize(existing.asset);
      existing.asset = mat;
      existing.refCount++;
    } else {
      _cache[mat.name] = _Entry(asset: mat, refCount: 1, persistent: true);
    }
    _currentBytes += _assetSize(mat);
    _evictIfNeeded();
    _safeEmit('loaded', mat.name);
  }

  /// Preload a batch: supports List<StaticAsset> and List<AssetConfig>
  Future<void> preloadBatch(List<dynamic> items) async {
    if (items.isEmpty) return;
    final first = items.first;
    if (first is StaticAsset) {
      for (final a in items.cast<StaticAsset>()) {
        await preload(a);
      }
      return;
    }
    if (first is AssetConfig) {
      for (final ref in items.cast<AssetConfig>()) {
        try {
          await load(ref);
        } catch (e) {
          _safeEmit('failed', ref.name, message: '$e');
          if (ref.required == true) rethrow;
        }
      }
      return;
    }
    throw ArgumentError(
      'Unsupported preloadBatch item type: ${first.runtimeType}',
    );
  }

  /// Get a cached asset by name.
  StaticAsset? get(String name) {
    if (_cache.containsKey(name)) _touch(name);
    return _cache[name]?.asset;
  }

  // Typed getters some stages use
  Uint8List? getPreloadedBinary(String name) {
    if (_cache.containsKey(name)) _touch(name);
    return _cache[name]?.asset.bytes;
  }

  Map<String, dynamic>? getPreloadedConfig(String name) {
    if (_cache.containsKey(name)) _touch(name);
    final a = _cache[name]?.asset;
    if (a == null) return null;

    if (a.text != null) {
      try {
        final v = json.decode(a.text!);
        return v is Map<String, dynamic>
            ? v
            : (v as Map).cast<String, dynamic>();
      } catch (_) {
        return null; // non-JSON text
      }
    }
    if (a.bytes != null) {
      try {
        final v = json.decode(utf8.decode(a.bytes!));
        return v is Map<String, dynamic>
            ? v
            : (v as Map).cast<String, dynamic>();
      } catch (_) {
        return null; // non-JSON bytes
      }
    }
    return null;
  }

  Tensor? getPreloadedTensor(String name) => _cache[name]?.asset.tensor;

  /// Load an asset from AssetConfig and cache it (persistent).
  Future<StaticAsset> load(AssetConfig ref) async {
    final existing = _cache[ref.name]?.asset;
    if (existing != null) {
      _cache[ref.name]!.refCount++;
      return existing;
    }

    final src = ref.source;
    final st = _parseSourceType(src['type'] as String?);
    StaticAsset asset;

    switch (st) {
      case AssetSourceType.file:
        final path = src['path'] as String?;
        if (path == null) {
          throw ArgumentError('Asset "${ref.name}" missing file path');
        }
        final bytes = await File(path).readAsBytes();
        final type = _inferType(ref.typeHint, src);
        asset = StaticAsset.bytes(
          name: ref.name,
          type: type,
          data: bytes,
          isRequired: ref.required,
        );
        break;
      case AssetSourceType.url:
        final url = src['url'] as String?;
        if (url == null) {
          throw ArgumentError('Asset "${ref.name}" missing url');
        }
        final client = HttpClient();
        try {
          final req = await client.getUrl(Uri.parse(url));
          final resp = await req.close();
          if (resp.statusCode != 200) {
            throw HttpException('HTTP ${resp.statusCode} for $url');
          }
          final bytes = await consolidateHttpClientResponseBytes(resp);
          final type = _inferType(ref.typeHint, src);
          asset = StaticAsset.bytes(
            name: ref.name,
            type: type,
            data: bytes,
            isRequired: ref.required,
          );
        } finally {
          client.close(force: true);
        }
        break;
      case AssetSourceType.embedded:
        final srcMap = src;
        // Support explicit 'text' or 'bytes' keys, else fallback to 'data'.
        if (srcMap.containsKey('text')) {
          final text = srcMap['text'] as String? ?? '';
          asset = StaticAsset.text(
            name: ref.name,
            data: text,
            isRequired: ref.required,
          );
        } else if (srcMap.containsKey('bytes')) {
          final raw = srcMap['bytes'];
          final bytes = raw is String ? _asBytes(raw) : _asBytes(raw);
          asset = StaticAsset.bytes(
            name: ref.name,
            type: AssetType.binary,
            data: bytes,
            isRequired: ref.required,
          );
        } else {
          final data = srcMap['data'];
          if (data == null) {
            throw ArgumentError('Asset "${ref.name}" missing embedded data');
          }
          final hint = ref.typeHint;
          if (data is Uint8List || data is List<int>) {
            final bytes = _asBytes(data);
            asset = StaticAsset.bytes(
              name: ref.name,
              type: AssetType.binary,
              data: bytes,
              isRequired: ref.required,
            );
          } else if (data is String) {
            // Decide text vs bytes using hint, JSON, base64 heuristics.
            if (hint != null &&
                (hint.toLowerCase() == 'text' ||
                    hint.toLowerCase() == 'json')) {
              asset = StaticAsset.text(
                name: ref.name,
                data: data,
                isRequired: ref.required,
              );
            } else if (hint != null && hint.toLowerCase() == 'binary') {
              asset = StaticAsset.bytes(
                name: ref.name,
                type: AssetType.binary,
                data: _asBytes(data),
                isRequired: ref.required,
              );
            } else if (_looksLikeJsonString(data)) {
              asset = StaticAsset.text(
                name: ref.name,
                data: data,
                isRequired: ref.required,
              );
            } else if (_looksLikeBase64(data)) {
              asset = StaticAsset.bytes(
                name: ref.name,
                type: AssetType.binary,
                data: _asBytes(data),
                isRequired: ref.required,
              );
            } else {
              // Default to text for plain strings.
              asset = StaticAsset.text(
                name: ref.name,
                data: data,
                isRequired: ref.required,
              );
            }
          } else {
            // Fallback: serialize to bytes if possible.
            final bytes = _asBytes(data);
            asset = StaticAsset.bytes(
              name: ref.name,
              type: AssetType.binary,
              data: bytes,
              isRequired: ref.required,
            );
          }
        }
        break;
      case AssetSourceType.generated:
        // Tests expect not implemented
        throw UnsupportedError(
          'generated source not implemented for "${ref.name}"',
        );
      case AssetSourceType.preloaded:
        final cached = _cache[ref.name]?.asset;
        if (cached == null) {
          throw StateError('preloaded asset "${ref.name}" not found');
        }
        asset = cached;
        break;
    }

    // Always cache loaded assets for tests
    _cache[ref.name] = _Entry(asset: asset, refCount: 1, persistent: true);
    _currentBytes += _assetSize(asset);
    _evictIfNeeded();
    _safeEmit('loaded', ref.name);
    return asset;
  }

  /// Release decreases refcount; evict if zero and not persistent.
  Future<void> release(String name) async {
    final entry = _cache[name];
    if (entry == null) return;
    entry.refCount = (entry.refCount - 1).clamp(0, 1 << 30);
    if (entry.refCount == 0 && !entry.persistent) {
      final removed = _cache.remove(name);
      if (removed != null) {
        _currentBytes -= _assetSize(removed.asset);
      }
      _safeEmit('evicted', name);
    } else {
      _safeEmit('released', name);
    }
  }

  DisposableToken registerPerCall(
    Map<String, StaticAsset> assets, {
    bool promoteToCache = false,
  }) {
    final names = <String>[];
    for (final e in assets.entries) {
      _cache[e.key] = _Entry(
        asset: e.value,
        refCount: 1,
        persistent: promoteToCache,
      );
      names.add(e.key);
      _safeEmit('loaded', e.key);
    }
    _evictIfNeeded();
    return DisposableToken(() async {
      for (final name in names) {
        final entry = _cache[name];
        if (entry == null) continue;
        entry.refCount = (entry.refCount - 1).clamp(0, 1 << 30);
        if (entry.refCount == 0 && !entry.persistent) {
          final removed = _cache.remove(name);
          if (removed != null) {
            _currentBytes -= _assetSize(removed.asset);
          }
          _safeEmit('evicted', name);
        } else {
          _safeEmit('released', name);
        }
      }
    });
  }

  // Materialize stage-provided asset if it references a source.
  Future<StaticAsset> _materializeIfNeeded(StaticAsset a) async {
    if (a.bytes != null || a.text != null || a.tensor != null) return a;
    final src = a.source;
    if (src == null) return a;

    final st = _parseSourceType(src['type'] as String?);
    switch (st) {
      case AssetSourceType.file:
        final path = src['path'] as String?;
        if (path == null) return a;
        final bytes = await File(path).readAsBytes();
        a.bytes = bytes;
        return a;
      case AssetSourceType.url:
        final url = src['url'] as String?;
        if (url == null) return a;
        final client = HttpClient();
        try {
          final req = await client.getUrl(Uri.parse(url));
          final resp = await req.close();
          if (resp.statusCode != 200) return a;
          final bytes = await consolidateHttpClientResponseBytes(resp);
          a.bytes = bytes;
        } finally {
          client.close(force: true);
        }
        return a;
      case AssetSourceType.embedded:
        // Support explicit 'text' or 'bytes' keys, else fallback to 'data'.
        if (src.containsKey('text')) {
          final text = src['text'] as String? ?? '';
          a.text = text;
          return a;
        }
        if (src.containsKey('bytes')) {
          final raw = src['bytes'];
          a.bytes = raw is String ? _asBytes(raw) : _asBytes(raw);
          return a;
        }
        final data = src['data'];
        if (data == null) return a;
        if (data is Uint8List || data is List<int>) {
          a.bytes = _asBytes(data);
          return a;
        }
        if (data is String) {
          if (_looksLikeJsonString(data)) {
            a.text = data;
          } else if (_looksLikeBase64(data)) {
            a.bytes = _asBytes(data);
          } else {
            a.text = data;
          }
          return a;
        }
        a.bytes = _asBytes(data);
        return a;
      case AssetSourceType.generated:
        // Keep unimplemented to align with tests elsewhere
        throw UnsupportedError('generated materialization not implemented');
      case AssetSourceType.preloaded:
        return a;
    }
  }
}

/// Helper to read all bytes from HttpClientResponse
Future<Uint8List> consolidateHttpClientResponseBytes(
  HttpClientResponse response,
) async {
  final chunks = <List<int>>[];
  int contentLength = 0;
  await for (final data in response) {
    chunks.add(data);
    contentLength += data.length;
  }
  final bytes = Uint8List(contentLength);
  int offset = 0;
  for (final chunk in chunks) {
    bytes.setRange(offset, offset + chunk.length, chunk);
    offset += chunk.length;
  }
  return bytes;
}
