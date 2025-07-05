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

    bool initializeModel();

public:
    OnnxProcessor();
    ~OnnxProcessor() = default;
    
    // Main processing function
    std::vector<float> processFrame(int width, int height, JNIEnv* env, jobject context, 
                                   const uint8_t* frameData, size_t frameSize);
};

} // namespace UniversalScanner