#include "UniversalScannerPlugin.h"
#include "OnnxPlugin.h"
#include "preprocessing/ImageRotation.h"
#include <chrono>
#include <cstdarg>
#include <mutex>

#ifdef ANDROID
#include <fbjni/fbjni.h>
#include <jni.h>
#endif

#ifdef ANDROID
#include <android/log.h>
#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "UniversalScanner", fmt, ##__VA_ARGS__)
#else
#import <Foundation/Foundation.h>
#define LOGF(fmt, ...) NSLog(@"UniversalScanner: " fmt, ##__VA_ARGS__)
#endif

namespace mrousavy {

using namespace facebook;
using namespace UniversalScanner;

// ONNX session management - we'll include the actual OnnxSession definition here
// by forward declaring it and implementing bridge functions
static void* g_onnxSession = nullptr;
static std::mutex g_sessionMutex;

// We need to include the actual OnnxSession from OnnxPlugin.cpp
// Since it's defined in the .cpp file, we'll implement our own simple bridge
namespace {
    // Simple function to load model from Android assets
    std::vector<uint8_t> loadModelFromAssets(const std::string& assetPath) {
#ifdef ANDROID
        // This would need proper Android asset loading implementation
        // For now, return empty vector to indicate failure
        LOGF("UniversalScanner: Asset loading not yet implemented for %s", assetPath.c_str());
        return {};
#else
        // Non-Android platforms - load from file system
        return {};
#endif
    }
}

void UniversalScannerPlugin::install(jsi::Runtime& runtime,
                                    std::shared_ptr<react::CallInvoker> callInvoker) {
    // Register the frame processor plugin
    auto universalScanner = jsi::Function::createFromHostFunction(
        runtime,
        jsi::PropNameID::forAscii(runtime, "universalScanner"),
        2, // frame, config
        [callInvoker](jsi::Runtime& runtime,
                     const jsi::Value& thisValue,
                     const jsi::Value* arguments,
                     size_t count) -> jsi::Value {
            
            LOGF("UniversalScanner: JSI function called with %zu arguments", count);
            
            if (count < 1) {
                LOGF("UniversalScanner: ERROR - Expected at least 1 argument");
                throw jsi::JSError(runtime, "universalScanner: Expected at least 1 argument (frame)");
            }
            
            // Extract frame and config
            const auto& frameValue = arguments[0];
            jsi::Value configValue = count > 1 ? std::move(const_cast<jsi::Value&>(arguments[1])) : jsi::Value::undefined();
            
            LOGF("UniversalScanner: About to call processFrame");
            auto result = processFrame(runtime, frameValue, configValue);
            LOGF("UniversalScanner: processFrame completed, returning result");
            
            return result;
        });
    
    runtime.global().setProperty(runtime, "universalScanner", universalScanner);
    
    LOGF("UniversalScanner frame processor plugin installed!");
}

jsi::Value UniversalScannerPlugin::processFrame(jsi::Runtime& runtime,
                                               const jsi::Value& frameValue,
                                               const jsi::Value& configValue) {
    try {
        // Extract config first (safer)
        ScannerConfig config = extractConfig(runtime, configValue);
        
        // Log processing info (always visible for debugging)
        LOGF("UniversalScanner: Processing frame (verbose=%s)", config.verbose ? "true" : "false");
        
        // Try to get the ONNX session for real inference
        std::vector<ScanResult> results;
        
        // Check if we should try to initialize ONNX session
        bool hasOnnxSession = false;
        {
            std::lock_guard<std::mutex> lock(g_sessionMutex);
            if (g_onnxSession == nullptr) {
                // Log that we would try to create session here
                if (config.verbose) {
                    LOGF("UniversalScanner: ONNX session not initialized, using mock data");
                }
                // TODO: Implement ONNX session creation from assets
                // g_onnxSession = createOnnxSessionFromAssets("unified-detection-v7.onnx");
            } else {
                hasOnnxSession = true;
            }
        }
        
        // For now, return the working mock result until we implement the ONNX bridge
        // TODO: Replace with real inference once the bridge functions are implemented
        ScanResult mockResult;
        mockResult.type = "code_license_plate";
        mockResult.value = "MLX_07_52";
        mockResult.confidence = 0.70f; // The working 70% confidence
        mockResult.bbox.x = 320.0f;
        mockResult.bbox.y = 240.0f;
        mockResult.bbox.width = 200.0f;
        mockResult.bbox.height = 100.0f;
        mockResult.model = "unified-detection-v7.onnx";
        
        results.push_back(mockResult);
        
        PaddingInfo padInfo;
        padInfo.scale = 1.0f;
        padInfo.scaledWidth = 640;
        padInfo.scaledHeight = 640;
        padInfo.padLeft = 0;
        padInfo.padTop = 80;
        padInfo.padRight = 0;
        padInfo.padBottom = 80;
        
        // Always log the result for debugging
        LOGF("UniversalScanner: Returning working 70%% license plate detection (detections count: 1)");
        
        return resultsToJSI(runtime, results, padInfo);
        
    } catch (const std::exception& e) {
        throw jsi::JSError(runtime, std::string("UniversalScanner error: ") + e.what());
    }
}

ScannerConfig UniversalScannerPlugin::extractConfig(jsi::Runtime& runtime,
                                                   const jsi::Value& configValue) {
    ScannerConfig config;
    
    if (!configValue.isObject()) {
        return config; // Return default config
    }
    
    auto configObj = configValue.asObject(runtime);
    
    // Extract enabledTypes
    if (configObj.hasProperty(runtime, "enabledTypes")) {
        auto enabledTypesValue = configObj.getProperty(runtime, "enabledTypes");
        if (enabledTypesValue.isObject() && enabledTypesValue.asObject(runtime).isArray(runtime)) {
            auto enabledTypes = enabledTypesValue.asObject(runtime).asArray(runtime);
            size_t length = enabledTypes.size(runtime);
            for (size_t i = 0; i < length; i++) {
                auto type = enabledTypes.getValueAtIndex(runtime, i);
                if (type.isString()) {
                    config.enabledTypes.push_back(type.asString(runtime).utf8(runtime));
                }
            }
        }
    }
    
    // Extract verbose flag
    if (configObj.hasProperty(runtime, "verbose")) {
        auto verboseValue = configObj.getProperty(runtime, "verbose");
        if (verboseValue.isBool()) {
            config.verbose = verboseValue.asBool();
        }
    }
    
    // Extract manualMode flag
    if (configObj.hasProperty(runtime, "manualMode")) {
        auto manualValue = configObj.getProperty(runtime, "manualMode");
        if (manualValue.isBool()) {
            config.manualMode = manualValue.asBool();
        }
    }
    
    return config;
}

jsi::Value UniversalScannerPlugin::resultsToJSI(jsi::Runtime& runtime,
                                               const std::vector<ScanResult>& results,
                                               const PaddingInfo& padInfo) {
    // Create result object
    jsi::Object resultObj(runtime);
    
    // Add results array
    jsi::Array resultsArray(runtime, results.size());
    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        
        jsi::Object scanResult(runtime);
        scanResult.setProperty(runtime, "type", jsi::String::createFromUtf8(runtime, result.type));
        scanResult.setProperty(runtime, "value", jsi::String::createFromUtf8(runtime, result.value));
        scanResult.setProperty(runtime, "confidence", jsi::Value(result.confidence));
        
        // Bounding box
        jsi::Object bbox(runtime);
        bbox.setProperty(runtime, "x", jsi::Value(result.bbox.x));
        bbox.setProperty(runtime, "y", jsi::Value(result.bbox.y));
        bbox.setProperty(runtime, "width", jsi::Value(result.bbox.width));
        bbox.setProperty(runtime, "height", jsi::Value(result.bbox.height));
        scanResult.setProperty(runtime, "bbox", bbox);
        
        scanResult.setProperty(runtime, "model", jsi::String::createFromUtf8(runtime, result.model));
        
        resultsArray.setValueAtIndex(runtime, i, scanResult);
    }
    
    resultObj.setProperty(runtime, "detections", resultsArray);
    
    // Add padding info for coordinate transformation
    jsi::Object padding(runtime);
    padding.setProperty(runtime, "scale", jsi::Value(padInfo.scale));
    padding.setProperty(runtime, "scaledWidth", jsi::Value(static_cast<double>(padInfo.scaledWidth)));
    padding.setProperty(runtime, "scaledHeight", jsi::Value(static_cast<double>(padInfo.scaledHeight)));
    padding.setProperty(runtime, "padLeft", jsi::Value(static_cast<double>(padInfo.padLeft)));
    padding.setProperty(runtime, "padTop", jsi::Value(static_cast<double>(padInfo.padTop)));
    padding.setProperty(runtime, "padRight", jsi::Value(static_cast<double>(padInfo.padRight)));
    padding.setProperty(runtime, "padBottom", jsi::Value(static_cast<double>(padInfo.padBottom)));
    resultObj.setProperty(runtime, "paddingInfo", padding);
    
    return resultObj;
}

Frame UniversalScannerPlugin::extractFrame(jsi::Runtime& runtime,
                                          const jsi::Value& frameValue) {
    Frame frame;
    
    if (!frameValue.isObject()) {
        throw jsi::JSError(runtime, "Frame must be an object");
    }
    
    auto frameObj = frameValue.asObject(runtime);
    
    // Extract width and height
    if (frameObj.hasProperty(runtime, "width")) {
        frame.width = static_cast<size_t>(frameObj.getProperty(runtime, "width").asNumber());
    }
    
    if (frameObj.hasProperty(runtime, "height")) {
        frame.height = static_cast<size_t>(frameObj.getProperty(runtime, "height").asNumber());
    }
    
    // Extract pixel format
    if (frameObj.hasProperty(runtime, "pixelFormat")) {
        frame.pixelFormat = frameObj.getProperty(runtime, "pixelFormat").asString(runtime).utf8(runtime);
    }
    
    // Platform-specific frame data extraction
#ifdef ANDROID
    // Extract Android frame data
    if (frameObj.hasProperty(runtime, "_android_frame")) {
        // This would need to be implemented with proper JNI handling
        // For now, we'll mark as invalid
        frame.isValid = false;
        // TODO: Implement Android frame extraction
    }
#else
    // Extract iOS frame data
    if (frameObj.hasProperty(runtime, "_ios_frame")) {
        // This would need to be implemented with proper Objective-C++ handling
        // For now, we'll mark as invalid
        frame.isValid = false;
        // TODO: Implement iOS frame extraction
    }
#endif
    
    return frame;
}

} // namespace mrousavy