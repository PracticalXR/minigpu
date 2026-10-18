import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:collection';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'asset_source_web.dart' if (dart.library.io) 'asset_source_io.dart';
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

  /// Whether a pipeline fails to start when this asset fails to load. A
  /// stage asset that is not required is skipped instead.
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

  /// An asset a generator would make. Nothing generates assets, so loading
  /// one fails with an [UnsupportedError].
  @Deprecated(
    'Nothing generates assets, so loading one always fails. Removed in 2.0.',
  )
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

  /// The bytes [asset] counts for against the cache limit.
  int size;

  /// How many holders the asset has: loads, preloads and per-call tokens
  /// that have not released it.
  int refCount = 1;

  /// Whether the asset stays cached when nothing holds it.
  bool persistent;

  _Entry(this.asset, this.size, {required this.persistent});
}

AssetSourceType? _parseSourceType(Object? s) => switch (s) {
  null || 'embedded' => AssetSourceType.embedded,
  'file' => AssetSourceType.file,
  'url' => AssetSourceType.url,
  'generated' => AssetSourceType.generated,
  'preloaded' => AssetSourceType.preloaded,
  _ => null,
};

AssetType? _hintType(String? hint) {
  switch (hint?.toLowerCase()) {
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
      return null;
  }
}

AssetType _inferType(String? hint, Map<String, dynamic> source) {
  final hinted = _hintType(hint);
  if (hinted != null) return hinted;

  // Infer from file extension
  final path = source['path'];
  if (path is String) {
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

  return AssetType.unknown;
}

/// Whether embedded `data` written as a string holds base64 bytes for an
/// asset of [type], rather than text.
bool _holdsBytes(AssetType? type) => switch (type) {
  AssetType.binary ||
  AssetType.weights ||
  AssetType.texture ||
  AssetType.lookupTable ||
  AssetType.tensor => true,
  _ => false,
};

final _whitespace = RegExp(r'\s');

Uint8List _bytesOf(String name, String key, Object? value) {
  if (value is String) {
    try {
      return base64.decode(value.replaceAll(_whitespace, ''));
    } on FormatException catch (e) {
      throw FormatException(
        'Asset "$name": "$key" is not valid base64 (${e.message})',
      );
    }
  }
  if (value is List && value.every((v) => v is int && v >= 0 && v < 256)) {
    return Uint8List.fromList(value.cast<int>());
  }
  throw FormatException(
    'Asset "$name": "$key" must be a list of byte values (0 to 255) or a '
    'base64 string',
  );
}

/// The data of the embedded asset source [source] of the asset [name]: a
/// [String] for text, or a [Uint8List] for bytes.
///
/// `text` is a string. `bytes` is a list of byte values or a base64 string.
/// `data` is a list of byte values, or a string: base64 when [typeHint] names
/// a binary type (`binary`, `weights`, `texture`, `lut` or `tensor`), and
/// text otherwise.
///
/// Throws a [FormatException] when the source has none of these keys, or a
/// value of another form.
Object readEmbeddedSource(
  String name,
  Map<String, dynamic> source,
  String? typeHint,
) {
  if (source.containsKey('text')) {
    final text = source['text'];
    if (text is String) return text;
    throw FormatException('Asset "$name": "text" must be a string');
  }
  if (source.containsKey('bytes')) {
    return _bytesOf(name, 'bytes', source['bytes']);
  }
  if (!source.containsKey('data')) {
    throw FormatException(
      'Asset "$name": an embedded source needs "text", "bytes" or "data"',
    );
  }
  final data = source['data'];
  if (data is String && !_holdsBytes(_hintType(typeHint))) return data;
  return _bytesOf(name, 'data', data);
}

/// Why the source of [config] cannot load, or null when it may. Only the
/// form of the source is checked: a file or url can still fail to read.
String? assetSourceProblem(AssetConfig config) {
  final src = config.source;
  final name = config.name;
  switch (_parseSourceType(src['type'])) {
    case AssetSourceType.file:
      if (src['path'] is String) return null;
      return 'Asset "$name": a file source needs a "path"';
    case AssetSourceType.url:
      if (src['url'] is String) return null;
      return 'Asset "$name": a url source needs a "url"';
    case AssetSourceType.embedded:
      try {
        readEmbeddedSource(name, src, config.typeHint);
        return null;
      } on FormatException catch (e) {
        return e.message;
      }
    case AssetSourceType.generated:
      return 'Asset "$name": generated sources are not implemented';
    case AssetSourceType.preloaded:
      return null;
    case null:
      return 'Asset "$name": unknown source type "${src['type']}"';
  }
}

class DisposableToken {
  final Future<void> Function() _dispose;
  DisposableToken(this._dispose);
  Future<void> dispose() => _dispose();
}

/// Loads, caches and counts the holders of the assets of one or more
/// pipelines.
///
/// [load], [preload] and [registerPerCall] each take a reference to the
/// asset they cache; [release] and disposing a per-call token drop one. An
/// asset that nothing holds any more is removed, unless it is persistent:
/// then it stays cached until the cache limit evicts it.
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

  /// The cache limit in bytes, or null when the cache is unlimited.
  int? get cacheLimit => _maxBytes;

  /// Sets the cache limit to [maxBytes]; 0 or less removes it. While the
  /// cache is over its limit, the least recently used assets that nothing
  /// holds are evicted.
  Future<void> setCacheLimit(int maxBytes) async {
    if (maxBytes <= 0) {
      _maxBytes = null; // disable limit
      return;
    }
    _maxBytes = maxBytes;
    _evictIfNeeded();
  }

  // The bytes an asset counts for: its bytes, or its text as UTF-8. A tensor
  // lives on the GPU and counts for nothing.
  static int _assetSize(StaticAsset a) {
    if (a.bytes != null) return a.bytes!.lengthInBytes;
    if (a.text != null) return utf8.encode(a.text!).length;
    return 0;
  }

  void _touch(String name) {
    final e = _cache.remove(name);
    if (e != null) _cache[name] = e;
  }

  _Entry _put(String name, StaticAsset asset, {required bool persistent}) {
    final entry = _Entry(asset, _assetSize(asset), persistent: persistent);
    _cache[name] = entry;
    _currentBytes += entry.size;
    return entry;
  }

  _Entry? _remove(String name) {
    final entry = _cache.remove(name);
    if (entry != null) _currentBytes -= entry.size;
    return entry;
  }

  // Adds a holder to [entry], cached as [name]. It becomes persistent if the
  // new holder asks for that.
  StaticAsset _acquire(String name, _Entry entry, {required bool persistent}) {
    entry.refCount++;
    entry.persistent = entry.persistent || persistent;
    _touch(name);
    return entry.asset;
  }

  // Drops a holder of [entry], cached as [name].
  void _drop(String name, _Entry entry) {
    if (entry.refCount > 0) entry.refCount--;
    if (entry.refCount == 0 && !entry.persistent) {
      _remove(name);
      _safeEmit('evicted', name);
    } else {
      _safeEmit('released', name);
      _evictIfNeeded();
    }
  }

  void _evictIfNeeded() {
    final limit = _maxBytes;
    if (limit == null || _currentBytes <= limit) return;
    // Oldest use first; an asset something holds is never evicted.
    final unheld = [
      for (final e in _cache.entries)
        if (e.value.refCount == 0) e.key,
    ];
    for (final name in unheld) {
      if (_currentBytes <= limit) break;
      _remove(name);
      _safeEmit('evicted', name);
    }
  }

  void clearAsset(String name) {
    if (_remove(name) != null) _safeEmit('evicted', name);
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

  /// Caches [asset] and takes a reference to it, first reading its data if
  /// it only names a source. A cached asset of the same name gains the
  /// reference and takes the new data. A preloaded asset is persistent.
  ///
  /// When reading the data fails, emits a `failed` event and rethrows.
  Future<void> preload(StaticAsset asset) async {
    final StaticAsset mat;
    try {
      mat = await _materializeIfNeeded(asset);
    } catch (e) {
      _safeEmit('failed', asset.name, message: '$e');
      rethrow;
    }
    final existing = _cache[mat.name];
    if (existing != null) {
      final size = _assetSize(mat);
      _currentBytes += size - existing.size;
      existing
        ..asset = mat
        ..size = size;
      _acquire(mat.name, existing, persistent: true);
    } else {
      _put(mat.name, mat, persistent: true);
    }
    _evictIfNeeded();
    _safeEmit('loaded', mat.name);
  }

  /// Loads each of [items]: a [StaticAsset] with [preload], an [AssetConfig]
  /// with [load].
  ///
  /// An item that is not required and fails to load is skipped; its `failed`
  /// event reports it. When a required one fails, the references this call
  /// took are released and the error is rethrown.
  Future<void> preloadBatch(List<dynamic> items) async {
    for (final item in items) {
      if (item is! StaticAsset && item is! AssetConfig) {
        throw ArgumentError(
          'Unsupported preloadBatch item type: ${item.runtimeType}',
        );
      }
    }
    final taken = <String>[];
    try {
      for (final item in items) {
        if (item is StaticAsset) {
          if (await _loadOrSkip(() => preload(item), item.isRequired)) {
            taken.add(item.name);
          }
        } else if (item is AssetConfig) {
          if (await _loadOrSkip(() => load(item), item.required)) {
            taken.add(item.name);
          }
        }
      }
    } catch (_) {
      for (final name in taken) {
        await release(name);
      }
      rethrow;
    }
  }

  // Whether [load] succeeded; a failure rethrows only when [required].
  static Future<bool> _loadOrSkip(
    Future<void> Function() load,
    bool required,
  ) async {
    try {
      await load();
      return true;
    } catch (_) {
      if (required) rethrow;
      return false;
    }
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

  Tensor? getPreloadedTensor(String name) {
    if (_cache.containsKey(name)) _touch(name);
    return _cache[name]?.asset.tensor;
  }

  /// Loads the asset [ref] describes, caches it and takes a reference to it.
  ///
  /// A cached asset of the same name is not read again; it gains the
  /// reference. When reading fails, emits a `failed` event and rethrows.
  /// The asset is persistent when [AssetConfig.cache] is true.
  Future<StaticAsset> load(AssetConfig ref) async {
    final cached = _cache[ref.name];
    if (cached != null) {
      return _acquire(ref.name, cached, persistent: ref.cache);
    }

    final StaticAsset asset;
    try {
      asset = await _read(ref);
    } catch (e) {
      _safeEmit('failed', ref.name, message: '$e');
      rethrow;
    }

    // Another load of this name may have finished while this one read.
    final raced = _cache[ref.name];
    if (raced != null) {
      return _acquire(ref.name, raced, persistent: ref.cache);
    }
    _put(ref.name, asset, persistent: ref.cache);
    _evictIfNeeded();
    _safeEmit('loaded', ref.name);
    return asset;
  }

  Future<StaticAsset> _read(AssetConfig ref) async {
    final src = ref.source;
    switch (_parseSourceType(src['type'])) {
      case AssetSourceType.file:
        final path = src['path'];
        if (path is! String) {
          throw ArgumentError('Asset "${ref.name}" has no file path');
        }
        return StaticAsset.bytes(
          name: ref.name,
          type: _inferType(ref.typeHint, src),
          data: await readFileSource(path),
          isRequired: ref.required,
        );
      case AssetSourceType.url:
        final url = src['url'];
        if (url is! String) {
          throw ArgumentError('Asset "${ref.name}" has no url');
        }
        return StaticAsset.bytes(
          name: ref.name,
          type: _inferType(ref.typeHint, src),
          data: await readUrlSource(url),
          isRequired: ref.required,
        );
      case AssetSourceType.embedded:
        final data = readEmbeddedSource(ref.name, src, ref.typeHint);
        return data is String
            ? StaticAsset.text(
                name: ref.name,
                data: data,
                isRequired: ref.required,
              )
            : StaticAsset.bytes(
                name: ref.name,
                type: AssetType.binary,
                data: data as Uint8List,
                isRequired: ref.required,
              );
      case AssetSourceType.generated:
        throw UnsupportedError(
          'Asset "${ref.name}": generated sources are not implemented',
        );
      case AssetSourceType.preloaded:
        throw StateError('preloaded asset "${ref.name}" not found');
      case null:
        throw ArgumentError(
          'Asset "${ref.name}" has an unknown source type "${src['type']}"',
        );
    }
  }

  /// Drops one holder of the asset [name]. When nothing holds it any more,
  /// a persistent asset stays cached (emitting `released`) until the cache
  /// limit evicts it, and any other is removed (emitting `evicted`).
  Future<void> release(String name) async {
    final entry = _cache[name];
    if (entry != null) _drop(name, entry);
  }

  /// Caches [assets] under their keys for a short time, such as one frame,
  /// and returns a token whose disposal drops them again.
  ///
  /// Throws a [StateError], and caches nothing, when a key names a cached
  /// asset that something holds. A cached asset that nothing holds is
  /// replaced. With [promoteToCache] the assets are persistent, so they stay
  /// cached after the token is disposed.
  DisposableToken registerPerCall(
    Map<String, StaticAsset> assets, {
    bool promoteToCache = false,
  }) {
    final inUse = [
      for (final name in assets.keys)
        if ((_cache[name]?.refCount ?? 0) > 0) name,
    ];
    if (inUse.isNotEmpty) {
      throw StateError(
        'Cannot register per-call assets $inUse: cached assets of those '
        'names are in use',
      );
    }
    final entries = <String, _Entry>{};
    for (final e in assets.entries) {
      if (_remove(e.key) != null) _safeEmit('evicted', e.key);
      entries[e.key] = _put(e.key, e.value, persistent: promoteToCache);
      _safeEmit('loaded', e.key);
    }
    _evictIfNeeded();
    var disposed = false;
    return DisposableToken(() async {
      if (disposed) return;
      disposed = true;
      for (final e in entries.entries) {
        // Only the entry this call cached: the name may have been cleared
        // and loaded again since.
        if (identical(_cache[e.key], e.value)) _drop(e.key, e.value);
      }
    });
  }

  // Reads the data of a stage-provided asset that only names its source.
  Future<StaticAsset> _materializeIfNeeded(StaticAsset a) async {
    if (a.bytes != null || a.text != null || a.tensor != null) return a;
    final src = a.source;
    if (src == null) return a;

    switch (_parseSourceType(src['type'])) {
      case AssetSourceType.file:
        final path = src['path'];
        if (path is! String) {
          throw ArgumentError('Asset "${a.name}" has no file path');
        }
        a.bytes = await readFileSource(path);
      case AssetSourceType.url:
        final url = src['url'];
        if (url is! String) {
          throw ArgumentError('Asset "${a.name}" has no url');
        }
        a.bytes = await readUrlSource(url);
      case AssetSourceType.embedded:
        final data = readEmbeddedSource(a.name, src, a.type.name);
        if (data is String) {
          a.text = data;
        } else {
          a.bytes = data as Uint8List;
        }
      case AssetSourceType.generated:
        throw UnsupportedError(
          'Asset "${a.name}": generated sources are not implemented',
        );
      case AssetSourceType.preloaded:
        break;
      case null:
        throw ArgumentError(
          'Asset "${a.name}" has an unknown source type "${src['type']}"',
        );
    }
    return a;
  }
}
