#pragma once

#include "../core/types.hpp"
#include <string>
#include <fstream>
#include <mutex>
#include <source_location>

namespace atom::logging {

enum class LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Critical = 5
};

class Logger {
public:
    static Logger& Instance();
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;
    
    // Configuration
    void SetLevel(LogLevel level) { min_level_ = level; }
    LogLevel GetLevel() const { return min_level_; }
    
    void SetLogFile(const std::string& filename);
    void EnableConsoleOutput(bool enable) { console_output_ = enable; }
    void EnableFileOutput(bool enable) { file_output_ = enable; }
    
    // Logging methods
    template<typename... Args>
    void Log(LogLevel level, const std::string& message, Args&&... args) {
        if (level < min_level_) return;
        
        std::string formatted = FormatMessage(message, std::forward<Args>(args)...);
        WriteLog(level, formatted, std::source_location::current());
    }
    
    template<typename... Args>
    void Trace(const std::string& message, Args&&... args) {
        Log(LogLevel::Trace, message, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void Debug(const std::string& message, Args&&... args) {
        Log(LogLevel::Debug, message, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void Info(const std::string& message, Args&&... args) {
        Log(LogLevel::Info, message, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void Warning(const std::string& message, Args&&... args) {
        Log(LogLevel::Warning, message, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void Error(const std::string& message, Args&&... args) {
        Log(LogLevel::Error, message, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void Critical(const std::string& message, Args&&... args) {
        Log(LogLevel::Critical, message, std::forward<Args>(args)...);
    }
    
    void Flush();
    
private:
    Logger() = default;
    ~Logger();
    
    LogLevel min_level_{LogLevel::Info};
    bool console_output_{true};
    bool file_output_{false};
    std::ofstream log_file_;
    mutable std::mutex mutex_;
    
    template<typename... Args>
    std::string FormatMessage(const std::string& format, Args&&... args) {
        if constexpr (sizeof...(args) == 0) {
            return format;
        } else {
            // Simple formatting - can be enhanced with std::format in C++20
            return format;
        }
    }
    
    void WriteLog(LogLevel level, const std::string& message, 
                  const std::source_location& location);
    std::string LevelToString(LogLevel level) const;
    std::string GetTimestamp() const;
};

// Convenience macros
#define LOG_TRACE(...) atom::logging::Logger::Instance().Trace(__VA_ARGS__)
#define LOG_DEBUG(...) atom::logging::Logger::Instance().Debug(__VA_ARGS__)
#define LOG_INFO(...) atom::logging::Logger::Instance().Info(__VA_ARGS__)
#define LOG_WARNING(...) atom::logging::Logger::Instance().Warning(__VA_ARGS__)
#define LOG_ERROR(...) atom::logging::Logger::Instance().Error(__VA_ARGS__)
#define LOG_CRITICAL(...) atom::logging::Logger::Instance().Critical(__VA_ARGS__)

} // namespace atom::logging
