# Set the shell explicitly for UNIX-like systems
SHELL := /bin/bash

# Detect the operating system
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	DEL_CMD := del /s /q
	RMDIR_CMD := rmdir /s /q
	MKDIR_CMD := mkdir
	SLASH := \\
else
	DETECTED_OS := $(shell uname)
	DEL_CMD := rm -rf
	RMDIR_CMD := rm -rf
	MKDIR_CMD := mkdir -p
	SLASH := /
endif

PLATFORM_INTERFACE_DIR := ./minigpu_platform_interface/
FFI_DIR := .$(SLASH)minigpu_ffi$(SLASH)
WEB_DIR := .$(SLASH)minigpu_web$(SLASH)
minigpu_DIR := .$(SLASH)minigpu$(SLASH)
EXAMPLE_DIR := .$(SLASH)minigpu$(SLASH)example$(SLASH)
SRC_DIR := .$(SLASH)minigpu_ffi$(SLASH)src$(SLASH)
BUILD_DIR := .$(SLASH)minigpu_ffi$(SLASH)src$(SLASH)external$(SLASH)dawn$(SLASH)
BUILD_WEB_DIR := .$(SLASH)minigpu_ffi$(SLASH)src$(SLASH)build_web$(SLASH)
VERSION ?= 1.0.0

default: run

pubspec_local:
	@echo "󰐊 Switching our pubspecs for local dev with version ${VERSION}."
	@python update_pubspecs.py ${VERSION}

pubspec_release:
	@echo "󰐊 Switching our pubspecs for release with version ${VERSION}."
	@python update_pubspecs.py ${VERSION} --release

clean:
	@echo "󰃢 Cleaning Example."
	@cd $(EXAMPLE_DIR) && flutter clean

run:
ifeq ($(DETECTED_OS), Windows)
	@echo "󰐊 Running example on Windows..."
	@cd $(EXAMPLE_DIR) && cmd /c flutter run -d Windows
else ifeq ($(DETECTED_OS), Linux)
	@echo "󰐊 Running example on $(DETECTED_OS)..."
	@cd $(EXAMPLE_DIR) && flutter run -d Linux
else ifeq ($(DETECTED_OS), Darwin)
	@echo "󰐊 Running example on $(DETECTED_OS)..."
	@cd $(EXAMPLE_DIR) && flutter run -d MacOS
else
	@echo "Unsupported OS: $(DETECTED_OS)"
endif

run_device:
	@echo "󰐊 Running example on device..."
	@cd $(EXAMPLE_DIR) && flutter run

run_web: build_weblib
	@echo "󰐊 Running web example..."
	@cd $(EXAMPLE_DIR) && flutter run -d chrome --web-browser-flag "--enable-features=SharedArrayBuffer" --web-browser-flag "--enable-unsafe-webgpu" --web-browser-flag "--disable-dawn-features=diallow_unsafe_apis"
ffigen:
	@echo "Generating dart ffi bindings..."
	@cd $(FFI_DIR) && dart run ffigen

build_weblib:
	@echo "Building ffi lib to web via emscripten..."
	@cd $(BUILD_WEB_DIR) && emcmake cmake .. -DCMAKE_TOOLCHAIN_FILE=$(EMSDK)/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake && cmake --build .

build_ffilib:
	@echo "Building ffi lib to native via emscripten..."
	@cd $(BUILD_DIR) && cmake cmake .. --fresh && cmake --build .

clean_weblib:
	@echo "Cleaning web lib dir..."
	@$(DEL_CMD) "$(BUILD_WEB_DIR)*"
	@$(RMDIR_CMD) "$(BUILD_WEB_DIR)*"

clean_ffilib:
	@echo "Cleaning lib dir..."
	@$(RMDIR_CMD) "$(BUILD_DIR)build_android"
	@$(RMDIR_CMD) "$(BUILD_DIR)build_win"
	@$(RMDIR_CMD) "$(BUILD_DIR)build_unix"

clean_weblib:
	@echo "Cleaning web lib dir..."
	@$(RMDIR_CMD) "$(BUILD_WEB_DIR)"
	@$(MKDIR_CMD) "$(BUILD_WEB_DIR)"

help:
	@echo ** Available Commands **
	@echo *  make pubspec_local: Switches pubspecs for local dev."
	@echo *  make pubspec_release: Switches pubspecs for release."
	@echo *  make clean: Cleans the example project."
	@echo *  make run: Runs the example project on current OS."
	@echo *  make run_device: Runs the example project on chosen device."
	@echo *  make run_web: Runs the web example project."
	@echo *  make ffigen: Generates dart ffi bindings."
	@echo *  make build_weblib: Builds the ffi lib to web via emscripten."
	@echo *  make build_ffilib: Builds the ffi lib to native."
	@echo *  make clean_weblib: Cleans the web lib."
	@echo *  make clean_ffilib: Cleans the native lib."
	@echo *  make help: Shows this help message."