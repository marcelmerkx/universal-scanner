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
    
    // Helper to check if detection is valid
    bool isValid() const { return confidence > 0.0f; }
};

class OnnxProcessor {
private:
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::Env> ortEnv;
    Ort::MemoryInfo memoryInfo;
    bool modelLoaded;
    
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

    bool initializeModel();

public:
    OnnxProcessor();
    ~OnnxProcessor() = default;
    
    // Main processing function
    DetectionResult processFrame(int width, int height, JNIEnv* env, jobject context, 
                                const uint8_t* frameData, size_t frameSize);
    
    // Debug image control
    void setDebugImages(bool enabled) { enableDebugImages = enabled; }
};

} // namespace UniversalScanner