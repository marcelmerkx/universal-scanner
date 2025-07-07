#pragma once

#include <chrono>
#include <string>

#ifdef ANDROID
#include <android/log.h>
#define PERF_LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "PerformanceTimer", fmt, ##__VA_ARGS__)
#else
#import <Foundation/Foundation.h>
#define PERF_LOGF(fmt, ...) NSLog(@"PerformanceTimer: " fmt, ##__VA_ARGS__)
#endif

namespace UniversalScanner {

/**
 * High-precision timing utility for performance profiling
 * Usage:
 *   PerformanceTimer timer("operation_name");
 *   // ... do work ...
 *   timer.logElapsed(); // Logs elapsed time
 */
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::string operationName;
    bool autoLog;
    
public:
    /**
     * Start timing an operation
     * @param name Operation name for logging
     * @param autoLogOnDestroy If true, automatically logs elapsed time on destruction
     */
    explicit PerformanceTimer(const std::string& name, bool autoLogOnDestroy = false) 
        : operationName(name), autoLog(autoLogOnDestroy) {
        start();
    }
    
    /**
     * Destructor - auto-logs if enabled
     */
    ~PerformanceTimer() {
        if (autoLog) {
            logElapsed();
        }
    }
    
    /**
     * Start/restart the timer
     */
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * Get elapsed time in milliseconds
     */
    double getElapsedMs() const {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        return duration.count() / 1000000.0; // Convert nanoseconds to milliseconds
    }
    
    /**
     * Get elapsed time in microseconds (for very fast operations)
     */
    double getElapsedUs() const {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        return duration.count() / 1000.0; // Convert nanoseconds to microseconds
    }
    
    /**
     * Log elapsed time with operation name
     */
    void logElapsed() const {
        double elapsedMs = getElapsedMs();
        PERF_LOGF("⏱️ %s: %.2f ms", operationName.c_str(), elapsedMs);
    }
    
    /**
     * Log elapsed time with custom message
     */
    void logElapsed(const std::string& message) const {
        double elapsedMs = getElapsedMs();
        PERF_LOGF("⏱️ %s (%s): %.2f ms", operationName.c_str(), message.c_str(), elapsedMs);
    }
    
    /**
     * Check if operation is taking longer than expected threshold
     */
    bool isSlowOperation(double thresholdMs = 33.0) const {
        return getElapsedMs() > thresholdMs;
    }
};

/**
 * Convenience macro for scoped timing with auto-logging
 * Usage: PERF_TIMER("operation_name");
 */
#define PERF_TIMER(name) UniversalScanner::PerformanceTimer _timer(name, true)

/**
 * Convenience macro for timing specific code blocks
 * Usage: PERF_MEASURE("operation") { // code }
 */
#define PERF_MEASURE(name) \
    for (UniversalScanner::PerformanceTimer _timer(name, true); _timer.getElapsedMs() >= 0; _timer.logElapsed(), break)

} // namespace UniversalScanner