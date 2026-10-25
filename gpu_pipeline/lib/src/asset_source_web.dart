// Reads `url` asset sources with fetch, which a page and a worker both have.
// Native builds use asset_source_io.dart instead.
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:web/web.dart' as web;

/// Files cannot be read on the web, so this always throws an
/// [UnsupportedError].
Future<Uint8List> readFileSource(String path) async => throw UnsupportedError(
  'Cannot read the file "$path" on the web; use a url or embedded source',
);

@JS('fetch')
external JSPromise<web.Response> _fetch(JSString url);

/// The body of a GET of [url]. Throws a [StateError] for any status but 200.
Future<Uint8List> readUrlSource(String url) async {
  final response = await _fetch(url.toJS).toDart;
  if (response.status != 200) {
    throw StateError('HTTP ${response.status} for $url');
  }
  final body = await response.arrayBuffer().toDart;
  return body.toDart.asUint8List();
}
