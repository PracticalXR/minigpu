# Assets

Assets are data that a pipeline loads once and its steps read, such as
weights, lookup tables and settings. Each pipeline has an `AssetManager`,
`pipeline.assetManager`. The pipelines of a network loaded from JSON share
one, `loaded.assets`. To share one between pipelines you build in code, pass
it as `Pipeline(assetManager: ...)`.

The examples on this page run as tests in `test/doc_examples_test.dart`, and
their `expect` lines show the results. `_tensor` and `_read` are the helpers
shown in [serialization.md](serialization.md).

## Assets in a JSON network

A network's `assets` load when one of its pipelines starts. This network
embeds three in the document:

```json
"assets": [
  {"name": "settings", "source": {"type": "embedded", "data": "{\"gain\": 3.0}"}},
  {"name": "table", "source": {"type": "embedded", "bytes": "AQIDBA=="}},
  {"name": "steps", "source": {"type": "embedded", "data": [5, 6, 7, 8]}}
],
```

```dart
final loaded = PipelineLoader().loadFromJson(_assetsJson);
final pipeline = loaded.runtime.pipelines.single;

await pipeline.start();
expect(loaded.assets.getPreloadedConfig('settings'), {'gain': 3.0});
expect(loaded.assets.getPreloadedBinary('table'), [1, 2, 3, 4]);
expect(loaded.assets.getPreloadedBinary('steps'), [5, 6, 7, 8]);

await loaded.dispose();
```

Keys marked * are required.

| Key | Meaning |
| --- | --- |
| `name`* | The name steps read the asset by. |
| `source`* | Where the data comes from; see below. |
| `required` | Default `false`. If a required asset fails to load, `start()` throws. An optional one emits a `failed` event and is skipped. |
| `strategy` | Default `preload`: the asset loads when a pipeline starts. The pipeline does not load `lazy`, `per_call` or `stream` assets; load them yourself with `loaded.assets.load(...)`, from `loaded.config.assets`. Any other value is a validation error. |
| `cache` | Whether the asset stays cached once nothing holds it; see [How long assets stay](#how-long-assets-stay). Default `true` for `preload` assets and `false` for the others. |
| `type_hint` | For embedded `data`, whether a string is base64; see below. For `file` and `url` sources it sets the asset's `type`, which otherwise comes from the file extension. |

`metadata` is read but not used.

| `source` | Data |
| --- | --- |
| `{"type": "file", "path": "..."}` | The file's bytes. |
| `{"type": "url", "url": "..."}` | The response body. Any status other than 200 fails. |
| `{"type": "embedded", "text": "..."}` | The string, as text. |
| `{"type": "embedded", "bytes": ...}` | Bytes, from a list of byte values (0 to 255) or a base64 string. |
| `{"type": "embedded", "data": ...}` | Bytes from a list of byte values. A string is text, unless `type_hint` names a binary type: `binary` (or `bin`), `weights`, `texture`, `lut` (or `lookuptable`) or `tensor`. Then it is base64. |
| `{"type": "preloaded"}` | The asset already cached under the name, for instance by `preload`. It fails if there is none. |
| `{"type": "generated"}` | Not implemented; it fails to load. |

A source without a `type` is embedded. Embedded text loads as a `text` asset
and embedded bytes as a `binary` one, whatever the hint.

An unknown `type` fails to load, and so does embedded data of another form:
`"text"` that is not a string, `"bytes"` that is not base64, a byte value
over 255, or a source with none of `text`, `bytes` and `data`. The validator
reports these, and a file or url source without its `path` or `url`, as
errors for a required asset and as warnings for an optional one.

## Stage assets

A stage built in Dart can declare its own assets with `StageBuilder`:

| Method | Asset |
| --- | --- |
| `withConfig(name, path: ...)` | A JSON file. |
| `withWeights`, `withTexture`, `withLookupTable` | A file's bytes, with the same arguments. |
| `withAsset(asset: ...)` | Any `StaticAsset`. `StaticAsset.text(name:, data:)`, `StaticAsset.bytes(name:, type:, data:)` and `StaticAsset.tensor(name:, data:)` hold their data in memory. |

They load into the pipeline's manager when the pipeline starts:

```dart
final pipeline = Pipeline(id: 'calibrated');
pipeline.addStage(
  StageBuilder('apply_gain')
      .withAsset(
        asset: StaticAsset.text(
          name: 'calibration',
          data: '{"gain": 3.0}',
        ),
      )
      .executeCPU(
        (inputs, ranks, params) async {
          final cfg = pipeline.assetManager.getPreloadedConfig(
            'calibration',
          )!;
          final gain = (cfg['gain'] as num).toDouble();
          final x = inputs['input']! as Float32List;
          return {
            'output': Float32List.fromList([for (final v in x) v * gain]),
          };
        },
        inputKey: 'input',
        outputKey: 'output',
      )
      .build(),
);

await pipeline.start(); // loads the stage's assets
final out = await pipeline.runOnce({
  'input': await _tensor([1, 2]),
});
expect(await _read(out['output']!), [3, 6]);

await pipeline.dispose();
```

If a required stage asset fails to load, `start()` throws; an optional one
emits a `failed` event and is skipped. The file methods make required assets
unless you pass `isRequired: false` (`required: false` for
`withLookupTable`). A custom stage can read the same manager through
`getAssetConfig`, `getAssetBinary` and `getAssetTensor`, which return null
until the pipeline has started.

`withGeneratedAsset` and `withGeneratedInput` are deprecated. Nothing
generates assets, so a required generated asset makes `start()` throw.

## Reading assets

| Call | Returns |
| --- | --- |
| `get(name)` | The `StaticAsset`, with its `text`, `bytes`, `tensor` and `type`, or null if nothing is loaded under that name. |
| `getPreloadedConfig(name)` | A text or bytes asset parsed as a JSON object, or null if it isn't one. |
| `getPreloadedBinary(name)` | A bytes asset's bytes, or null for a text asset. |
| `getPreloadedTensor(name)` | A tensor asset's tensor. |

There is no getter for text; use `get(name)?.text`.

## When assets load

`runOnce` does not load assets. `start()` loads the network's `preload`
assets, then each stage's assets, then initialises the stages and starts the
pipeline's streams. If a required asset fails to load, `start()` releases the
assets it loaded, emits a `PipelineErrorEvent` and rethrows the error.

The pipeline holds the assets it loaded until `stop()`, which releases them.
`dispose()` stops the pipeline first. A network asset that is already cached
is not loaded again: the cached asset gains a holder instead, so the
pipelines of a network share one copy of it. A stage asset whose name is
already cached replaces the cached asset's data.

## How long assets stay

Each load, preload and per-call token holds the asset it cached until
`release(name)` or disposing the token lets go. When nothing holds an asset
any more:

- a persistent asset stays cached and the manager emits `released`. Network
  assets with `cache: true`, preloaded assets (stage assets included) and
  per-call assets added with `promoteToCache: true` are persistent.
- any other asset is removed and the manager emits `evicted`.

A persistent asset that nothing holds stays until the cache limit evicts it
or you remove it.

## Loading and removing by hand

- `load(config)` loads an `AssetConfig` the way a network does, and returns
  the asset.
- `preload(asset)` caches a `StaticAsset`, first reading the source it names
  if it holds no data.
- `preloadBatch(items)` loads a list of both. It skips an optional item that
  fails to load; when a required one fails, it releases what it loaded and
  rethrows.
- `release(name)` lets go of one hold.
- `clearAsset(name)` removes one asset and `clearCache()` removes them all,
  held or not. Both emit `evicted`.
- `dispose()` clears the cache and closes `events`.

## Per-call assets

`registerPerCall` adds assets for a short time, such as one frame's metadata,
and returns a token. Disposing the token removes them:

```dart
final assets = AssetManager();
final seen = <String>[];
assets.events.listen((e) => seen.add('${e.kind} ${e.name}'));

final token = assets.registerPerCall({
  'frame_info': StaticAsset.text(name: 'frame_info', data: '{"n": 7}'),
});
expect(assets.getPreloadedConfig('frame_info'), {'n': 7});

await token.dispose();
expect(assets.get('frame_info'), isNull);
expect(seen, ['loaded frame_info', 'evicted frame_info']);

await assets.dispose();
```

If one of the names is cached and held, such as an asset of a running
pipeline, `registerPerCall` throws a `StateError` and adds nothing. A cached
asset that nothing holds is replaced. With `promoteToCache: true` the assets
are persistent, so they stay cached when the token is disposed. A token
removes only the assets it added: if a name was cleared and cached again in
the meantime, the new asset stays. Disposing a token twice does nothing.

## Events

`events` is a broadcast stream of `AssetEvent`s, each with a `kind`, the
asset's `name` and sometimes a `message`. The kinds are `loaded`, `failed`,
`released` and `evicted`. Events are delivered synchronously, during the call
that causes them.

## Cache limit

`Pipeline(maxMemoryMB: ...)` sets the manager's limit. A manager the pipeline
creates gets 4096 MB by default; a manager passed in keeps its own limit unless
`maxMemoryMB` is given. `setCacheLimit(bytes)` sets the limit directly, 0 or
less removes it, and `cacheLimit` reads it.

An asset counts for its bytes, or for its text as UTF-8. A tensor asset lives
on the GPU and counts for nothing. While the cache is over its limit, the
least recently used assets that nothing holds are evicted, each with an
`evicted` event; reading an asset counts as using it. An asset that something
holds is never evicted, so the cache can stay over its limit while a pipeline
runs.

## Lifetimes

`pipeline.dispose()` releases the pipeline's assets but leaves the asset
manager alone, since other pipelines may share it. `loaded.dispose()`
disposes a loaded network's pipelines, then its manager. Dispose a manager
you share yourself, after the last pipeline that uses it.

## On the web

`file` sources, and stage assets read from a `path`, fail on the web. `url`
sources are read with `fetch`, so the server must allow the page's origin
(CORS). Embedded data and `StaticAsset.text`, `StaticAsset.bytes` and
`StaticAsset.tensor` work everywhere, and so does data you fetch yourself and
add with `preload` or `registerPerCall`.
