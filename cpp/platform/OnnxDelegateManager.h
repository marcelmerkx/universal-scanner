#pragma once

#include <string>
#include <unordered_map>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>

#ifdef __ANDROID__
#include "nnapi_provider_factory.h"
#endif

#ifdef ANDROID
#include <android/log.h>
#define DELEGATE_LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "OnnxDelegateManager", fmt, ##__VA_ARGS__)
#else
#import <Foundation/Foundation.h>
#define DELEGATE_LOGF(fmt, ...) NSLog(@"OnnxDelegateManager: " fmt, ##__VA_ARGS__)
#endif

namespace UniversalScanner {

/**
 * Enum representing different ONNX Runtime execution providers
 */
enum class ExecutionProvider {
    CPU,
    XNNPACK,
    NNAPI,
    COREML
};

/**
 * Convert ExecutionProvider enum to string for logging
 */
inline const char* toString(ExecutionProvider ep) {
    switch (ep) {
        case ExecutionProvider::CPU: return "cpu";
        case ExecutionProvider::XNNPACK: return "xnnpack";
        case ExecutionProvider::NNAPI: return "nnapi";
        case ExecutionProvider::COREML: return "coreml";
        default: return "unknown";
    }
}

/**
 * Manager class for selecting and configuring the best available ONNX Runtime execution provider
 * 
 * Attempts to use hardware acceleration when available:
 * - Android: XNNPACK (optimized CPU) > NNAPI (GPU/NPU) > CPU
 * - iOS: CoreML (Neural Engine) > CPU
 * 
 * Gracefully handles initialization failures and falls back to CPU.
 */
class OnnxDelegateManager {
public:
    /**
     * Configure session options with the best available execution provider
     * 
     * @param sessionOptions ONNX Runtime session options to configure
     * @param verbose Enable verbose logging of provider selection
     * @param preferXNNPACK Prefer XNNPACK over NNAPI on Android (recommended)
     * @return ExecutionProvider that was selected and applied
     */
    static ExecutionProvider configure(Ort::SessionOptions& sessionOptions, bool verbose = false, bool preferXNNPACK = true, bool disableNNAPI = true) {
        // Set common optimization options
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.EnableMemPattern();
        
        // Use 4 threads for optimal mobile performance
        sessionOptions.SetInterOpNumThreads(4);
        sessionOptions.SetIntraOpNumThreads(4);
        
#ifdef __ANDROID__
        // Note: XNNPACK is not available in standard ONNX Runtime Android builds
        // We'll use the optimized CPU provider which includes ARM NEON optimizations
        
        // Try NNAPI as second option (unless disabled)
        if (!disableNNAPI) {
            try {
            // Use NNAPI with optimized flags
            uint32_t nnapi_flags = 0;
            nnapi_flags |= NNAPI_FLAG_USE_FP16;  // FP16 for performance
            nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;  // Disable inefficient CPU fallback
            
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, nnapi_flags);
            
            if (status == nullptr) {
                if (verbose) {
                    DELEGATE_LOGF("🚀 Using NNAPI provider (Android GPU/NPU acceleration)");
                }
                return ExecutionProvider::NNAPI;
            } else {
                // Release status if error occurred
                const OrtApi& ortApi = Ort::GetApi();
                ortApi.ReleaseStatus(status);
                throw std::runtime_error("NNAPI provider failed to initialize");
            }
            } catch (const std::exception& e) {
                if (verbose) {
                    DELEGATE_LOGF("⚠️ NNAPI unavailable (%s), falling back to CPU", e.what());
                }
            }
        } else if (verbose) {
            DELEGATE_LOGF("⏭️ NNAPI disabled by configuration");
        }
#endif

#ifdef __APPLE__
        // Try CoreML on iOS (Neural Engine acceleration)
        try {
            std::unordered_map<std::string, std::string> coreml_options;
            sessionOptions.AppendExecutionProvider("CoreMLExecutionProvider", coreml_options);
            if (verbose) {
                DELEGATE_LOGF("🚀 Using CoreML provider (iOS Neural Engine acceleration)");
            }
            return ExecutionProvider::COREML;
        } catch (const Ort::Exception& e) {
            if (verbose) {
                DELEGATE_LOGF("⚠️ CoreML unavailable (%s), falling back to CPU", e.what());
            }
        }
#endif

        // CPU fallback (available on all platforms)
        if (verbose) {
            DELEGATE_LOGF("🔧 Using optimized CPU provider");
        }
        return ExecutionProvider::CPU;
    }
    
    /**
     * Get human-readable description of execution provider performance characteristics
     */
    static const char* getPerformanceDescription(ExecutionProvider ep) {
        switch (ep) {
            case ExecutionProvider::CPU: 
                return "CPU execution (optimized kernels)";
            case ExecutionProvider::XNNPACK:
                return "XNNPACK acceleration (ARM SIMD optimized)";
            case ExecutionProvider::NNAPI: 
                return "NNAPI acceleration (Android GPU/NPU)";
            case ExecutionProvider::COREML: 
                return "CoreML acceleration (iOS Neural Engine)";
            default: 
                return "Unknown execution provider";
        }
    }
    
    /**
     * Estimate relative performance multiplier compared to basic CPU
     * These are rough estimates - actual performance depends on model and hardware
     */
    static float getPerformanceMultiplier(ExecutionProvider ep) {
        switch (ep) {
            case ExecutionProvider::CPU: 
                return 1.5f; // Optimized CPU kernels
            case ExecutionProvider::XNNPACK:
                return 2.5f; // Typically 2-3x faster than basic CPU
            case ExecutionProvider::NNAPI: 
                return 3.0f; // When it works well
            case ExecutionProvider::COREML: 
                return 4.0f; // Typically 3-6x faster than CPU
            default: 
                return 1.0f;
        }
    }
};

} // namespace UniversalScanner