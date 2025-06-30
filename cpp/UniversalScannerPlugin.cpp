#include "UniversalScannerPlugin.h"
#include "OnnxPlugin.h"
#include "preprocessing/ImageRotation.h"
#include <chrono>
#include <cstdarg>

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

// Forward declaration - the ONNX plugin will need to expose a global session or similar mechanism
// For now, we'll use a simplified approach

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
            
            if (count < 1) {
                throw jsi::JSError(runtime, "universalScanner: Expected at least 1 argument (frame)");
            }
            
            // Extract frame and config
            const auto& frameValue = arguments[0];
            jsi::Value configValue = count > 1 ? std::move(const_cast<jsi::Value&>(arguments[1])) : jsi::Value::undefined();
            
            return processFrame(runtime, frameValue, configValue);
        });
    
    runtime.global().setProperty(runtime, "universalScanner", universalScanner);
    
    LOGF("UniversalScanner frame processor plugin installed!");
}

jsi::Value UniversalScannerPlugin::processFrame(jsi::Runtime& runtime,
                                               const jsi::Value& frameValue,
                                               const jsi::Value& configValue) {
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Extract frame and config
        Frame frame = extractFrame(runtime, frameValue);
        ScannerConfig config = extractConfig(runtime, configValue);
        
        if (!frame.isValid) {
            throw jsi::JSError(runtime, "Invalid frame");
        }
        
        // TODO: Integration with ONNX model
        // For now, we'll just do the preprocessing pipeline
        
        // Step 1: YUV to RGB conversion
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> rgbData = FrameConverter::convertYUVtoRGB(frame);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Step 2: Rotation if needed
        size_t width = frame.width;
        size_t height = frame.height;
        
        if (ImageRotation::needsRotation(width, height)) {
            rgbData = ImageRotation::rotate90CCW(rgbData, width, height);
            std::swap(width, height);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // Step 3: White padding + normalization
        PaddingInfo padInfo;
        std::vector<float> tensorData = WhitePadding::applyPadding(
            rgbData, width, height, 640, &padInfo
        );
        auto t4 = std::chrono::high_resolution_clock::now();
        
        // Step 4: Run ONNX inference (TODO: integrate with loaded model)
        // For now, just measure the preprocessing time
        auto t5 = std::chrono::high_resolution_clock::now();
        
        // Log performance metrics if verbose
        if (config.verbose) {
            LOGF("Native preprocessing times (ms):");
            LOGF("  YUV→RGB: %.2f", std::chrono::duration<double, std::milli>(t2 - t1).count());
            LOGF("  Rotation: %.2f", std::chrono::duration<double, std::milli>(t3 - t2).count());
            LOGF("  Padding: %.2f", std::chrono::duration<double, std::milli>(t4 - t3).count());
            LOGF("  Inference: %.2f", std::chrono::duration<double, std::milli>(t5 - t4).count());
            LOGF("  Total: %.2f", std::chrono::duration<double, std::milli>(t5 - start).count());
        }
        
        // Parse ONNX output and create results
        std::vector<ScanResult> results;
        // TODO: Parse YOLO output format and create ScanResult objects
        
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
    
    resultObj.setProperty(runtime, "results", resultsArray);
    
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