# Vendored Dart API dynamic linking headers

Verbatim copies of `include/` from the Dart SDK
(`dart_api.h`, `dart_api_dl.c`, `dart_api_dl.h`, `dart_native_api.h`,
`dart_version.h`, `internal/dart_api_dl_impl.h`).

Copyright the Dart project authors; BSD-style license (header retained in
every file).

They are vendored so this native build can call `Dart_PostCObject_DL` and
deliver `mgpu::Logger` output to a Dart **native port** instead of an
`MGPULogCallback` function pointer. The log registry is process-global: a
`NativeCallable` left behind by an exited isolate aborts the VM the moment a
Dawn worker thread logs; posting to a closed port is defined, silent and
thread-safe.

Native builds only — a wasm module has no Dart VM to post to, so
`CMakeLists.txt` compiles `dart_api_dl.c` and defines `MINIGPU_HAVE_DART_DL`
only when `NOT EMSCRIPTEN`.

Update procedure: re-copy from `<dart-sdk>/include` — do not hand-edit.
