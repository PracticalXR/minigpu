# minigpu

## 1.2.0-WIP

## 1.1.9

- fix pubspec version issue
## 1.1.8

- fixed concurrent buffer op crash
## 1.1.6

- fixed problem with audio input capture providing raw data

## 1.1.5

- Refactored example
- Various fixes
- Updated native assets to code assets
- Memory problems fixed on web and ffi

## 1.1.3

- breaking: import package instead of buffer and shader separately
- fix: pubspec repository url
- adds: tensor package protoype
- fix: issue with reading buffer segments fixed

## 1.1.2

- fix: create dawn dir to prevent first run error.

## 1.1.1

- fix: split download command for quiet fail on remote add

## 1.1.0

- fix: dawn git not running properly

## 1.0.9

- fix: prevent using project root on ffi since pub wont see the file

## 1.0.8

- fix: pub.dev still missing project root file

## 1.0.7

- fix: project root file missing

## 1.0.6

- fix: minigpu_ffi must also use flutter in pubspec or pub.dev analysis fails
- fix: issue with project root finding as package

## 1.0.5

- fix: must have flutter in pubspec or pub.dev analysis fails

## 1.0.4

- fix: updates to readme

## 1.0.3

- fix: remove flutter from package pubspec.yaml
- fix: updates to readme

## 1.0.2

- new: explicity set supported platforms in pubspec.yaml for pub.dev

## 1.0.1

- breaking: Uses dart native assets
see updated readme.
- implements platform stub for native assets to coexist with flutter plugins.
- uses native_toolchain_cmake 0.0.4

## 1.0.0

- Initial version.
