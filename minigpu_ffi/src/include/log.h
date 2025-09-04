#pragma once

#include <iostream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cstring>

// Prefer std::format or {fmt} if available
#if defined(__has_include)
#  if __has_include(<format>) && defined(__cpp_lib_format)
#    include <format>
#    define MGPU_USE_STD_FORMAT 1
#  elif __has_include(<fmt/format.h>)
#    include <fmt/format.h>
#    define MGPU_USE_FMTLIB 1
#  endif
#endif

#include "../include/mutex.h"

namespace mgpu {

enum LogLevel {
    LOG_NONE = -1,
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
};

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

    template<typename... Args>
    void log(LogLevel level, const char* file, int line, const char* format, Args... args) {
        if (level_ == LOG_NONE || level < level_) return;

        mgpu::lock_guard<mgpu::mutex> lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        
        const char* levelStr[] = {"DEBUG", "INFO", "WARN", "ERROR"};
        const char* filename = strrchr(file, '/');
        if (!filename) filename = strrchr(file, '\\');
        if (!filename) filename = file;
        else filename++;

        // Build the message without triggering -Wformat-security
        std::string message;
    #if defined(MGPU_USE_STD_FORMAT)
        message = std::vformat(format, std::make_format_args(args...));
    #elif defined(MGPU_USE_FMTLIB)
        // fmt::runtime allows non-literal format strings safely
        message = fmt::format(fmt::runtime(format), args...);
    #else
        // Fallback to snprintf with size probing; suppress format-security warning locally
    #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wformat-security"
    #endif
        int n = std::snprintf(nullptr, 0, format, args...);
        if (n > 0) {
            std::vector<char> buf(static_cast<size_t>(n) + 1);
            std::snprintf(buf.data(), buf.size(), format, args...);
            message.assign(buf.data(), static_cast<size_t>(n));
        } else {
            message = format ? format : "";
        }
    #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
    #endif
    #endif

        std::string logLine = ss.str() + " [" + levelStr[static_cast<int>(level)] + "] " + 
                              filename + ":" + std::to_string(line) + " " + message;

        std::cout << logLine << std::endl;
    }

private:
    Logger() : level_(LOG_INFO) {}
    ~Logger() = default;

    LogLevel level_;
    mgpu::mutex mutex_;
};

} // namespace mgpu

// Convenience macros
#define LOG_DEBUG(...) mgpu::Logger::getInstance().log(mgpu::LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  mgpu::Logger::getInstance().log(mgpu::LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  mgpu::Logger::getInstance().log(mgpu::LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) mgpu::Logger::getInstance().log(mgpu::LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// Helper to set log level
#define SET_LOG_LEVEL(level) mgpu::Logger::getInstance().setLevel(level)