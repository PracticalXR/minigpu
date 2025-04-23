import 'dart:typed_data';

import 'gpu_tensor_base.dart';
import 'gpu_data.dart';

extension TensorPrintHelpers<T extends TypedData> on Tensor<T> {
  /// Returns a string representation of the first [counts] elements along each dimension.
  /// If [pretty] is true, the output is formatted with newlines and indentation.
  Future<String> head(List<int> counts, {bool pretty = false}) async {
    if (counts.length != shape.length) {
      throw Exception(
          "Counts length (${counts.length}) does not match tensor rank (${shape.length}).");
    }
    if (shape.length == 2) {
      int numRows = counts[0];
      int numCols = counts[1];
      int totalCols = shape[1];
      List<List<double>> rows = [];
      for (int r = 0; r < numRows; r++) {
        // For 2D printing, we always read a Float32List and then convert its values.
        final rowData = Float32List(numCols);
        // Calculate the flat offset for this row.
        await buffer.read(rowData, numCols, readOffset: r * totalCols);
        rows.add(rowData.map((e) => e.toDouble()).toList());
      }
      return _format2D(rows, pretty: pretty);
    } else {
      // Use slicing for higher dimensional tensors.
      List<int> startIndices = List.filled(shape.length, 0);
      Tensor<T> subTensor =
          await slice(startIndices: startIndices, endIndices: counts);
      T subData = await subTensor.getData();
      List<double> dataList = _toDoubleList(subData);
      return _formatTensor(dataList, counts, pretty: pretty);
    }
  }

  /// Returns a string representation of the last [counts] elements along each dimension.
  /// If [pretty] is true, the output is formatted with newlines and indentation.
  Future<String> tail(List<int> counts, {bool pretty = false}) async {
    if (counts.length != shape.length) {
      throw Exception(
          "Counts length (${counts.length}) does not match tensor rank (${shape.length}).");
    }
    if (shape.length == 2) {
      int numRows = counts[0];
      int numCols = counts[1];
      int totalCols = shape[1];
      int startRow = shape[0] - numRows;
      int startCol = totalCols - numCols;
      List<List<double>> rows = [];
      for (int r = startRow; r < shape[0]; r++) {
        final rowData = Float32List(numCols);
        await buffer.read(rowData, numCols,
            readOffset: r * totalCols + startCol);
        rows.add(rowData.map((e) => e.toDouble()).toList());
      }
      return _format2D(rows, pretty: pretty);
    } else {
      List<int> startIndices = [];
      for (int i = 0; i < shape.length; i++) {
        if (counts[i] > shape[i]) {
          throw Exception(
              "Tail count (${counts[i]}) exceeds tensor dimension $i size (${shape[i]}).");
        }
        startIndices.add(shape[i] - counts[i]);
      }
      Tensor<T> subTensor = await slice(
          startIndices: startIndices, endIndices: List<int>.from(shape));
      // For formatted output, slice the head of the sub-tensor.
      subTensor = await subTensor.slice(
          startIndices: List.filled(counts.length, 0), endIndices: counts);
      T subData = await subTensor.getData();
      List<double> dataList = _toDoubleList(subData);
      return _formatTensor(dataList, counts, pretty: pretty);
    }
  }

  /// Helper: converts [data] of type T to a List<double>.
  List<double> _toDoubleList(T data) {
    if (data is Float32List) {
      return data.map((e) => e.toDouble()).toList();
    } else if (data is Float64List) {
      return data.map((e) => e.toDouble()).toList();
    } else if (data is Int32List) {
      return data.map((e) => e.toDouble()).toList();
    } else if (data is Uint8List) {
      return data.map((e) => e.toDouble()).toList();
    } else {
      throw Exception("Unsupported TypedData type: ${data.runtimeType}");
    }
  }

  /// Helper for formatting a 2D list.
  String _format2D(List<List<double>> rows, {bool pretty = false}) {
    if (pretty) {
      StringBuffer sb = StringBuffer();
      sb.writeln('[');
      for (var row in rows) {
        sb.writeln('  ${row.toString()},');
      }
      sb.write(']');
      return sb.toString();
    } else {
      List<String> rowStrings = rows.map((row) => row.toString()).toList();
      return "[${rowStrings.join(", ")}]";
    }
  }

  /// Recursively formats a flat [data] list into a nested array string given [shape].
  /// [indent] controls the indentation level for pretty printing.
  String _formatTensor(List<double> data, List<int> shape,
      {bool pretty = false, int indent = 0}) {
    if (shape.length == 1) {
      return data.toString();
    } else {
      int subTensorSize = shape.sublist(1).reduce((a, b) => a * b);
      List<String> parts = [];
      String indentStr = pretty ? ' ' * indent : '';
      for (int i = 0; i < shape[0]; i++) {
        int startIdx = i * subTensorSize;
        int endIdx = startIdx + subTensorSize;
        List<double> subData = data.sublist(startIdx, endIdx);
        String subStr = _formatTensor(subData, shape.sublist(1),
            pretty: pretty, indent: indent + 2);
        parts.add(subStr);
      }
      if (pretty) {
        return '$indentStr[\n${parts.map((p) => '$indentStr  $p').join(',\n')}\n$indentStr]';
      } else {
        return "[${parts.join(", ")}]";
      }
    }
  }
}
