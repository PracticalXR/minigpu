#pragma once

#include <iostream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

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
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    template<typename... Args>
    void log(LogLevel level, const char* file, int line, const char* format, Args... args) {
        if (level_ == LOG_NONE || level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        
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

        char buffer[1024];
        snprintf(buffer, sizeof(buffer), format, args...);

        std::string logLine = ss.str() + " [" + levelStr[level] + "] " + 
                             filename + ":" + std::to_string(line) + " " + buffer;

        std::cout << logLine << std::endl;
    }

private:
    Logger() : level_(LOG_INFO) {}
    ~Logger() = default;

    LogLevel level_;
    std::mutex mutex_;
};

} // namespace mgpu

// Convenience macros
#define LOG_DEBUG(...) mgpu::Logger::getInstance().log(mgpu::LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) mgpu::Logger::getInstance().log(mgpu::LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) mgpu::Logger::getInstance().log(mgpu::LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) mgpu::Logger::getInstance().log(mgpu::LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// Helper to set log level
#define SET_LOG_LEVEL(level) mgpu::Logger::getInstance().setLevel(level)