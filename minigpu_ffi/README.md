# Minigpu FFI

This is the FFI implementation of the minigpu package.

see https://pub.dev/packages/minigpu

## Dawn build location

minigpu depends on [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU
implementation). On first build Dawn is cloned and compiled automatically.

| Platform | Default path                  |
|----------|-------------------------------|
| Windows  | `%SYSTEMDRIVE%\dawn` (e.g. `C:\dawn`) |
| macOS    | `~/dawn`                      |
| Linux    | `~/dawn`                      |

Windows uses a short root-level path to stay well within the 260-character
`MAX_PATH` limit that Dawn's deeply-nested source tree can otherwise exceed.

### Override: `MINIGPU_DAWN_DIR`

Set the `MINIGPU_DAWN_DIR` environment variable to point Dawn at a different
root directory. The build expects a `build_{os}_{arch}/` subdirectory inside
it (e.g. `build_win_x86_64/`).

```powershell
# Windows example — use a custom drive/path
$env:MINIGPU_DAWN_DIR = 'D:\my_dawn'
```

```bash
# macOS / Linux example
export MINIGPU_DAWN_DIR=/opt/dawn
```

This env var is read by both the Dart build hook (`hook/build.dart`) and the
CMake module (`src/cmake/dawn.cmake`), so it works whether you build via
`flutter build`, `dart pub get`, or `cmake` directly.