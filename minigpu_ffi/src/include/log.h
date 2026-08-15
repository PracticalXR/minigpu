#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "../include/mutex.h"

namespace mgpu {

enum LogLevel {
    LOG_NONE = -1,
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
};

/// Signature for a non-Dart embedder's log bridge.
/// level: 0=DEBUG 1=INFO 2=WARN 3=ERROR
///
/// DART EMBEDDERS MUST NOT USE THIS. The registry below is PROCESS-GLOBAL,
/// so a Dart `NativeCallable` installed here outlives the isolate that
/// created it: once that isolate exits the VM has deleted the trampoline
/// while this library still holds the pointer, and the next log line from a
/// Dawn worker thread aborts the entire process
/// ("Callback invoked after it has been deleted", isolate_group=(nil)).
/// Use the native-port path (`mgpuSetLogPort`) instead — see below.
using LogCallback = void(*)(int level, const char* message);

/// Dart native-port delivery, defined in minigpu.cpp (it owns the vendored
/// Dart API headers). `logPostToPort` returns true when a port is registered
/// AND the message was handed to the VM — in that case the function-pointer /
/// stderr paths are skipped. Always false in builds without MINIGPU_HAVE_DART_DL
/// (e.g. the Emscripten/web build), so behaviour there is unchanged.
bool logPostToPort(int level, const char* message, size_t len);
intptr_t logInitDartApi(void* initialize_api_dl_data);
void logSetPort(int64_t port);

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(LogLevel level) {
        mgpu::lock_guard<mgpu::mutex> lock(mutex_);
        level_ = level;
    }

    /// Install a callback that receives every log message (after level filter).
    /// Pass nullptr to revert to the default stderr output.
    void setCallback(LogCallback cb) {
        mgpu::lock_guard<mgpu::mutex> lock(mutex_);
        callback_ = cb;
    }

    template<typename... Args>
    void log(LogLevel level, const char* file, int line, const char* format, Args... args) {
        if (level_ == LOG_NONE || level < level_) return;

        mgpu::lock_guard<mgpu::mutex> lock(mutex_);

        const char* levelStr[] = {"DEBUG", "INFO", "WARN", "ERROR"};
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), format, args...);

        // A registered native port wins: it is the only Dart delivery path
        // that survives the registering isolate exiting.  Nothing is heap
        // allocated — Dart_PostCObject_DL copies the bytes into the message.
        if (logPostToPort(static_cast<int>(level), buffer, strlen(buffer))) return;

        if (callback_) {
            // NativeCallable.listener dispatches asynchronously on the Dart
            // event loop.  By the time Dart runs, this stack frame has
            // returned and `buffer` is gone.  Heap-copy so the pointer
            // stays valid; Dart must call mgpuFreeLogMessage() after reading.
            char* heap_msg = static_cast<char*>(malloc(strlen(buffer) + 1));
            if (heap_msg) {
                memcpy(heap_msg, buffer, strlen(buffer) + 1);
                callback_(static_cast<int>(level), heap_msg);
            }
        } else {
            const char* filename = strrchr(file, '/');
            if (!filename) filename = strrchr(file, '\\');
            if (!filename) filename = file;
            else filename++;
            std::fprintf(stderr, "[mgpu %s] %s:%d %s\n",
                levelStr[level], filename, line, buffer);
        }
    }

    /// Log a pre-formatted message at the given level (no file/line decoration).
    /// Used by the MGPU_LOG macro for raw fprintf-style messages.
    void logRaw(LogLevel level, const char* message) {
        if (level_ == LOG_NONE || level < level_) return;
        mgpu::lock_guard<mgpu::mutex> lock(mutex_);
        // Native port first — see log() above.
        if (logPostToPort(static_cast<int>(level), message, strlen(message))) return;
        if (callback_) {
            // Same heap-copy requirement as log() — see comment above.
            char* heap_msg = static_cast<char*>(malloc(strlen(message) + 1));
            if (heap_msg) {
                memcpy(heap_msg, message, strlen(message) + 1);
                callback_(static_cast<int>(level), heap_msg);
            }
        } else {
            std::fprintf(stderr, "%s\n", message);
        }
    }

private:
    Logger() : level_(LOG_INFO), callback_(nullptr) {}
    ~Logger() = default;

    LogLevel level_;
    LogCallback callback_;
    mgpu::mutex mutex_;
};

} // namespace mgpu

// Convenience macros
#define LOG_DEBUG(...) mgpu::Logger::getInstance().log(mgpu::LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) mgpu::Logger::getInstance().log(mgpu::LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) mgpu::Logger::getInstance().log(mgpu::LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) mgpu::Logger::getInstance().log(mgpu::LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// Helper to set log level
#define SET_LOG_LEVEL(level) mgpu::Logger::getInstance().setLevel(level)
#define SET_LOG_CALLBACK(cb) mgpu::Logger::getInstance().setCallback(cb)

// MGPU_LOG — format a message into a stack buffer then forward to Logger::logRaw.
// Use this in place of std::fprintf(stderr, "[mgpu ...] fmt", ...) so all
// messages pass through the single Logger (and its optional Dart callback).
// Level: 0=DEBUG 1=INFO 2=WARN 3=ERROR
#define MGPU_LOG(lvl, ...) \
    do { \
        char _mgpu_buf[1024]; \
        snprintf(_mgpu_buf, sizeof(_mgpu_buf), __VA_ARGS__); \
        mgpu::Logger::getInstance().logRaw( \
            static_cast<mgpu::LogLevel>(lvl), _mgpu_buf); \
    } while (0)