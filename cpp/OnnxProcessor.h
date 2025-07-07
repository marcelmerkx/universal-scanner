#pragma once

#include <vector>
#include <memory>
#include <string>
#include <jni.h>

// Include ONNX Runtime
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

// Include preprocessing modules
#include "preprocessing/FrameConverter.h"
#include "preprocessing/ImageRotation.h"
#include "preprocessing/WhitePadding.h"
#include "preprocessing/ImageDebugger.h"
#include "platform/YuvConverter.h"
#include "platform/YuvResizer.h"
#include "platform/OnnxDelegateManager.h"
#include "CodeDetectionConstants.h"
#include "utils/PerformanceTimer.h"

namespace UniversalScanner {

/**
 * Single detection result from ONNX inference
 * Coordinates are normalized to [0,1] range for display independence
 */
struct DetectionResult {
    float confidence;    // Detection confidence [0,1]
    float centerX;       // Center X coordinate normalized [0,1] 
    float centerY;       // Center Y coordinate normalized [0,1]
    float width;         // Bounding box width normalized [0,1]
    float height;        // Bounding box height normalized [0,1]
    int classIndex;      // Class index (0-4)
    bool hasDetection;   // Whether a detection was found
    
    // Helper to check if detection is valid
    bool isValid() const { return hasDetection && confidence > 0.0f; }
};

class OnnxProcessor {
private:
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::Env> ortEnv;
    Ort::MemoryInfo memoryInfo;
    bool modelLoaded;
    int modelSize_;  // Model input size (320, 416, or 640)
    
    struct ModelInfo {
        std::vector<int64_t> inputShape;
        std::vector<int64_t> outputShape;
        std::string inputName;
        std::string outputName;
    } modelInfo;

    // Platform-specific YUV converter and resizer
    std::unique_ptr<YuvConverter> yuvConverter;
    std::unique_ptr<IYuvResizer> yuvResizer;
    
    // Debug image logging control
    bool enableDebugImages;
    
    // Execution provider tracking for performance analysis
    ExecutionProvider currentExecutionProvider;
    
    // Android AssetManager for loading models
    jobject assetManager_;
    JNIEnv* env_;

    bool initializeModel();
    bool initializeConverters(JNIEnv* env, jobject context);
    
    std::vector<uint8_t> preprocessFrame(const uint8_t* frameData, size_t frameSize, 
                                        int width, int height, int* outWidth, int* outHeight);
    std::vector<float> createTensorFromRGB(const std::vector<uint8_t>& rgbData, 
                                          int width, int height);
    DetectionResult runInference(const std::vector<float>& inputTensor, uint8_t enabledCodeTypesMask);
    DetectionResult findBestDetection(const std::vector<float>& modelOutput, uint8_t enabledCodeTypesMask);

public:
    OnnxProcessor();
    ~OnnxProcessor() = default;
    
    // Set model size (will reload model if size changes)
    void setModelSize(int size);
    int getModelSize() const { return modelSize_; }
    
    // Main processing function
    DetectionResult processFrame(int width, int height, JNIEnv* env, jobject context, 
                                const uint8_t* frameData, size_t frameSize, uint8_t enabledCodeTypesMask = CodeDetectionMask::ALL);
    
    // Debug image control
    void setDebugImages(bool enabled) { enableDebugImages = enabled; }
    
    // Performance monitoring
    ExecutionProvider getExecutionProvider() const { return currentExecutionProvider; }
    const char* getExecutionProviderName() const { return toString(currentExecutionProvider); }
};

} // namespace UniversalScanner