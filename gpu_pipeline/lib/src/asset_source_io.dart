// Reads `file` and `url` asset sources with dart:io. The web build uses
// asset_source_web.dart instead.
import 'dart:io';
import 'dart:typed_data';

/// The bytes of the file at [path].
Future<Uint8List> readFileSource(String path) => File(path).readAsBytes();

/// The body of a GET of [url]. Throws an [HttpException] for any status but
/// 200.
Future<Uint8List> readUrlSource(String url) async {
  final uri = Uri.parse(url);
  final client = HttpClient();
  try {
    final response = await (await client.getUrl(uri)).close();
    if (response.statusCode != 200) {
      await response.drain<void>();
      throw HttpException('HTTP ${response.statusCode} for $url', uri: uri);
    }
    final body = BytesBuilder(copy: false);
    await response.forEach(body.add);
    return body.takeBytes();
  } finally {
    client.close(force: true);
  }
}
