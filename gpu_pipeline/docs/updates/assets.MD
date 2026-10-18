# Assets v1 — model, API, and usage

This document reflects the current AssetManager/StaticAsset implementation and how to use it in code and JSON.

What this gives you

- A single AssetManager that caches assets by logical name.
- Two ways to describe an asset:
  - StaticAsset: stage-provided, already-known at build time.
  - AssetRef (AssetConfig in Dart): serialized reference (file/url/embedded/...).
- Deterministic lifecycle with events and per-call (ephemeral) assets.
- Typed getters used by stages: getPreloadedBinary/getPreloadedConfig/getPreloadedTensor.

## Types

- AssetType: json, text, binary, weights, texture, tensor, lookupTable, shader, unknown
- AssetSourceType: file, url, embedded, generated, preloaded
- AssetLoadingStrategy: preload, lazy, perCall, stream (advisory)
- Events: AssetEvent(kind: 'loaded' | 'failed' | 'released' | 'evicted', name, message?)

## StaticAsset (stage-provided) factories

- StaticAsset.weights(name, path: ...)
- StaticAsset.texture(name: ..., path: ...)
- StaticAsset.config(name: ..., path: ...)  // json/text file
- StaticAsset.lookupTable(name: ..., path: ...)
- StaticAsset.generated(name: ..., type: ..., generatorParams: {...})  // stage-provided; load-time “generated source” from JSON is not implemented
- StaticAsset.bytes(name: ..., type: AssetType.binary, data: Uint8List)
- StaticAsset.text(name: ..., data: '...')
- StaticAsset.tensor(name: ..., data: Tensor)

## AssetRef JSON schema (snake_case)

Use this in pipeline/stage config JSON. In Dart tests you build AssetConfig directly.

- name: string
- source:
  - type: "file" | "url" | "embedded" | "generated" | "preloaded"
  - path?: string
  - url?: string
  - data?: any (inline)
- required?: bool (default false) — rethrows in preloadBatch/load if unavailable
- strategy?: "preload" | "lazy" | "per_call" | "stream" (advisory; manager preloads by default)
- cache?: bool (default true) — keep in cache
- type_hint?: string (e.g., "json", "weights", "shader")

## Examples

### File (weights)

```json
{
  "name": "mobilenet_v2",
  "source": { "type": "file", "path": "models/mnet_v2.bin" },
  "required": true,
  "strategy": "preload",
  "cache": true,
  "type_hint": "weights"
}
```

### Embedded (text)

```json
{
  "name": "hello",
  "source": { "type": "embedded", "data": "hello world" },
  "type_hint": "text"
}
```

### Embedded (binary via base64)

```json
{
  "name": "lut",
  "source": { "type": "embedded", "data": "AQIDBA==" }
}
```

### URL

```json
{
  "name": "remote_cfg",
  "source": { "type": "url", "url": "https://example.com/cfg.json" },
  "type_hint": "json",
  "required": false
}
```

Note: generated source in JSON is not implemented and will throw UnsupportedError.
4 AssetManager API (Dart)

### Construction and events

```dart
final assets = AssetManager();
final sub = assets.events.listen((e) {
  print('${e.kind} ${e.name} ${e.message ?? ""}');
});
```

### Preload stage-provided assets

```dart
await assets.preload(
  StaticAsset.weights('stage_weights', path: 'models/w.bin'),
);
// or batch
await assets.preloadBatch([
  StaticAsset.texture(name: 'bg', path: 'assets/bg.png'),
]);
```

### Preload from serialized refs (AssetConfig)

```dart
await assets.preloadBatch([
  AssetConfig(
    name: 'cfg',
    source: {'type': 'file', 'path': 'config/app.json'},
    typeHint: 'json',
    required: true, // will throw + emit failed on missing
  ),
]);
```

### On-demand load (caches on success)

```dart
final a = await assets.load(AssetConfig(
  name: 'txt',
  source: {'type': 'embedded', 'data': 'hello'},
  typeHint: 'text',
));
print(a.text); // 'hello'
```

### Typed getters used by stages

```dart
final bytes = assets.getPreloadedBinary('lut');      // Uint8List?
final map = assets.getPreloadedConfig('cfg_json');   // Map<String,dynamic>?
final t = assets.getPreloadedTensor('pre_tensor');   // Tensor?
```

### Per-call (ephemeral) assets

```dart
final token = assets.registerPerCall({
  'frame_image': StaticAsset.bytes(
    name: 'frame_image',
    type: AssetType.binary,
    data: Uint8List.fromList([1,2,3]),
  ),
});
// use assets during a single run...
await token.dispose(); // evicts unless promoted via promoteToCache flag
```

### Cache maintenance

```dart
assets.clearAsset('name'); // emits evicted
assets.clearCache();       // emits evicted for all
await assets.dispose();    // closes event stream and clears cache
```

## Runtime behavior and heuristics

- preloadBatch(List<AssetConfig>)
  - Loads each; on error emits failed event.
  - If ref.required == true, rethrows to caller.
  - Optional assets (required == false) do not throw; caller can check manager.get(name).

- Embedded payloads
  - source.text -> text
  - source.bytes -> binary
  - source.data string:
    - JSON-looking string -> text/json
    - Base64-looking string -> binary
    - Otherwise -> text
  - source.data List<int> / Uint8List -> binary

- URL
  - HTTP != 200 or connection error: emits failed and does not cache. If required, load/preloadBatch may throw.

- generated source (from JSON)
  - Not implemented; throws UnsupportedError.

## Pipeline integration

- Inject a shared AssetManager when constructing a Pipeline (hot-swaps keep the cache):

```dart
final assets = AssetManager();
final pipeline = Pipeline(id: 'p1', assetManager: assets);
```

- Stage-provided assets:
  - StageBuilder should add StaticAsset.* to the stage’s requiredAssets.
  - Pipeline.start() calls:
    - manager.preloadBatch(stage.requiredAssets)
    - stage.preloadAssets(manager)
  - Stages can retrieve assets by name using the typed getters.

- If you build pipelines from serialized configs, preload pipeline/stage AssetConfig refs before or during start:

```dart
await assets.preloadBatch(pipelineConfig.assets + stageConfig.assets);
```

## Quick examples

### Embedded JSON + binary

```dart
final m = AssetManager();
await m.preloadBatch([
  AssetConfig(
    name: 'cfg_json',
    source: {'type': 'embedded', 'data': '{"a":1}'},
    typeHint: 'json',
  ),
  AssetConfig(
    name: 'lut_bin',
    source: {'type': 'embedded', 'data': 'AQIDBA=='}, // base64
  ),
]);
final cfg = m.getPreloadedConfig('cfg_json')!;
final lut = m.getPreloadedBinary('lut_bin')!;
```

### Stage-provided weights file

```dart
await m.preload(StaticAsset.weights('w', path: 'models/w.bin'));
final w = m.get('w')!.bytes;
```

### Per-call frame data

```dart
final token = m.registerPerCall({
  'frame': StaticAsset.bytes(
    name: 'frame',
    type: AssetType.binary,
    data: frameBytes,
  ),
});
// use it...
await token.dispose();
```

## Event patterns

```dart
final events = <AssetEvent>[];
final sub = m.events.listen(events.add);

// after operations...
final failed = events.where((e) => e.kind == 'failed');
```

### Kinds you’ll see

- 'loaded'   when an asset is cached
- 'failed'   when a load/preload fails
- 'released' when release() decrements without eviction
- 'evicted'  when removed via clearAsset/clearCache or per-call dispose

## FAQ

- Do I need one AssetManager per pipeline?
  Prefer one shared instance per logical workspace/app so hot-swaps reuse the cache. Inject it into all Pipeline constructors.

- Why do embedded base64 strings become binary?
  Heuristics detect base64-like content; use type_hint: 'text' to force text if needed.

- Can I use generated assets from JSON?
  Not yet; stage-provided StaticAsset.generated is supported via code, but JSON source.type == 'generated' throws.

- How do stages access assets?
  Through AssetManager passed to stage.preloadAssets/initialize/process and via typed getters like getPreloadedBinary.
