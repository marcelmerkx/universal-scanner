#include "OnnxProcessor.h"
#include <android/log.h>

#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "UniversalScanner", fmt, ##__VA_ARGS__)

// Use getClassName from Universal.cpp to avoid duplicate symbol
extern const char* getClassName(int classIdx);

namespace UniversalScanner {

OnnxProcessor::OnnxProcessor() 
    : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), modelLoaded(false), yuvConverter(nullptr), enableDebugImages(false) {
    LOGF("OnnxProcessor created");
    
    // Enable debug images by default in DEBUG builds, disable in release
    #ifdef DEBUG
        enableDebugImages = true;
        LOGF("Debug images enabled (DEBUG build)");
    #else
        enableDebugImages = false;
        LOGF("Debug images disabled (RELEASE build)");
    #endif
    
    // Initialize model info for unified-detection-v7.onnx
    modelInfo.inputShape = {1, 3, 640, 640}; // NCHW format
    modelInfo.outputShape = {1, 9, 8400}; // 9 features x 8400 anchors (4 bbox + 5 classes, NO objectness)
    modelInfo.inputName = "images";
    modelInfo.outputName = "output0";
}

// Load model from file path 
std::vector<uint8_t> loadModelFromFile(const std::string& filePath) {
    FILE* file = fopen(filePath.c_str(), "rb");
    if (!file) {
        LOGF("Failed to open model file: %s", filePath.c_str());
        return {};
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read file data
    std::vector<uint8_t> modelData(fileSize);
    size_t bytesRead = fread(modelData.data(), 1, fileSize, file);
    fclose(file);
    
    if (bytesRead != fileSize) {
        LOGF("Failed to read complete model file");
        return {};
    }
    
    LOGF("Loaded ONNX model from file: %zu bytes", modelData.size());
    return modelData;
}

bool OnnxProcessor::initializeModel() {
    if (modelLoaded) return true;
    
    try {
        // Try to load model from internal storage
        std::string modelPath = "/data/data/com.cargosnap.universalscanner/files/unified-detection-v7.onnx";
        auto modelData = loadModelFromFile(modelPath);
        if (modelData.empty()) {
            // Try assets location
            modelPath = "/android_asset/unified-detection-v7.onnx";
            modelData = loadModelFromFile(modelPath);
            if (modelData.empty()) {
                LOGF("Failed to load model from any location");
                return false;
            }
        }
        
        // Create ONNX Runtime environment
        ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "UniversalScanner");
        
        // Create session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        
        // Create session from memory buffer
        session = std::make_unique<Ort::Session>(*ortEnv, modelData.data(), modelData.size(), sessionOptions);
        
        modelLoaded = true;
        LOGF("ONNX model loaded successfully!");
        return true;
        
    } catch (const Ort::Exception& e) {
        LOGF("Failed to load ONNX model: %s", e.what());
        return false;
    }
}

// [The processFrame method implementation would go here - it's very long]
// For now I'll add a stub and we can move the implementation

DetectionResult OnnxProcessor::processFrame(int width, int height, JNIEnv* env, jobject context, 
                                            const uint8_t* frameData, size_t frameSize, uint8_t enabledCodeTypesMask) {
    try {
        // Initialize components
        if (!modelLoaded && !initializeModel()) return {};
        if (!initializeConverters(env, context)) return {};
        
        // Preprocess frame: YUV resize + RGB conversion + rotation + padding
        int processedWidth, processedHeight;
        auto rgbData = preprocessFrame(frameData, frameSize, width, height, &processedWidth, &processedHeight);
        if (rgbData.empty()) return {};
        
        // Create ONNX tensor from preprocessed image
        auto inputTensor = createTensorFromRGB(rgbData, processedWidth, processedHeight);
        if (inputTensor.empty()) return {};
        
        // Run inference and return result
        return runInference(inputTensor, enabledCodeTypesMask);
        
    } catch (const Ort::Exception& e) {
        LOGF("ONNX inference error: %s", e.what());
        return {}; // NO FALLBACK MOCKING
    }
}

// Helper method implementations

bool OnnxProcessor::initializeConverters(JNIEnv* env, jobject context) {
    // Initialize YUV converter if not already done
    if (!yuvConverter && env) {
        LOGF("Initializing platform-specific YUV converter...");
        yuvConverter = YuvConverter::create(env, context);
        if (!yuvConverter) {
            LOGF("Failed to create YUV converter");
            return false;
        }
    }
    
    // Initialize YUV resizer if not already done  
    if (!yuvResizer && env) {
        LOGF("Initializing platform-specific YUV resizer...");
        yuvResizer = YuvResizer::create(env, context);
        if (!yuvResizer) {
            LOGF("Failed to create YUV resizer");
            return false;
        }
    }
    
    return yuvConverter != nullptr;
}

std::vector<uint8_t> OnnxProcessor::preprocessFrame(const uint8_t* frameData, size_t frameSize, 
                                                   int width, int height, int* outWidth, int* outHeight) {
    if (!frameData || frameSize == 0) {
        LOGF("ERROR: No real frame data provided");
        return {};
    }
    
    LOGF("Processing REAL camera frame data: %dx%d, %zu bytes", width, height, frameSize);
    
    // DEBUG: Save original YUV frame
    if (enableDebugImages) {
        size_t ySize = width * height;
        size_t uvSize = ySize / 4;
        const uint8_t* yPlane = frameData;
        const uint8_t* uPlane = frameData + ySize;
        const uint8_t* vPlane = frameData + ySize + uvSize;
        UniversalScanner::ImageDebugger::saveYUV420("0_original_yuv.jpg", 
            yPlane, uPlane, vPlane, width, height, width, width / 2);
    }
    
    // Step 1: Resize YUV for performance optimization
    int targetWidth, targetHeight;
    if (width > height) {
        targetWidth = 640;
        targetHeight = (height * 640) / width;
    } else {
        targetHeight = 640;
        targetWidth = (width * 640) / height;
    }
    
    // Ensure even dimensions for YUV 4:2:0 format
    targetWidth = targetWidth & 0xFFFE;
    targetHeight = targetHeight & 0xFFFE;
    
    std::vector<uint8_t> resizedYuvData;
    int processWidth = width;
    int processHeight = height;
    const uint8_t* processFrameData = frameData;
    
    if (yuvResizer && (targetWidth < width || targetHeight < height)) {
        LOGF("Resizing YUV: %dx%d -> %dx%d", width, height, targetWidth, targetHeight);
        resizedYuvData = yuvResizer->resizeYuv(frameData, frameSize, width, height, targetWidth, targetHeight);
        
        if (!resizedYuvData.empty()) {
            processWidth = targetWidth;
            processHeight = targetHeight;
            processFrameData = resizedYuvData.data();
            
            // DEBUG: Save resized YUV frame
            if (enableDebugImages) {
                size_t resizedYSize = processWidth * processHeight;
                size_t resizedUvSize = resizedYSize / 4;
                const uint8_t* resizedYPlane = processFrameData;
                const uint8_t* resizedUPlane = processFrameData + resizedYSize;
                const uint8_t* resizedVPlane = processFrameData + resizedYSize + resizedUvSize;
                UniversalScanner::ImageDebugger::saveYUV420("0b_resized_yuv.jpg", 
                    resizedYPlane, resizedUPlane, resizedVPlane, 
                    processWidth, processHeight, processWidth, processWidth / 2);
            }
        }
    }
    
    // Step 2: Convert YUV to RGB
    LOGF("Converting YUV to RGB (%dx%d)", processWidth, processHeight);
    size_t processFrameSize = resizedYuvData.empty() ? frameSize : resizedYuvData.size();
    auto rgbData = yuvConverter->convertYuvToRgb(processFrameData, processFrameSize, processWidth, processHeight);
    
    if (rgbData.empty()) {
        LOGF("ERROR: YUV to RGB conversion failed");
        return {};
    }
    
    if (enableDebugImages) {
        UniversalScanner::ImageDebugger::saveRGB("1_rgb_converted.jpg", rgbData, processWidth, processHeight);
    }
    
    // Step 3: Apply 90° CW rotation
    LOGF("Applying 90° CW rotation (%dx%d)", processWidth, processHeight);
    rgbData = UniversalScanner::ImageRotation::rotate90CW(rgbData, processWidth, processHeight);
    std::swap(processWidth, processHeight); // Dimensions swapped after rotation
    
    if (enableDebugImages) {
        UniversalScanner::ImageDebugger::saveRGB("2_rotated.jpg", rgbData, processWidth, processHeight);
    }
    
    *outWidth = processWidth;
    *outHeight = processHeight;
    return rgbData;
}

std::vector<float> OnnxProcessor::createTensorFromRGB(const std::vector<uint8_t>& rgbData, int width, int height) {
    LOGF("Creating tensor from RGB data (%dx%d)", width, height);
    
    // Apply white padding to make it square 640x640
    UniversalScanner::PaddingInfo padInfo;
    auto inputTensor = UniversalScanner::WhitePadding::applyPadding(rgbData, width, height, 640, &padInfo);
    
    if (inputTensor.empty()) {
        LOGF("ERROR: White padding failed");
        return {};
    }
    
    if (enableDebugImages) {
        UniversalScanner::ImageDebugger::saveTensor("3_padded.jpg", inputTensor, 640, 640);
    }
    
    LOGF("Tensor created: 640x640, first pixels: %.3f %.3f %.3f", 
         inputTensor[0], inputTensor[1], inputTensor[2]);
    
    return inputTensor;
}

DetectionResult OnnxProcessor::runInference(const std::vector<float>& inputTensor, uint8_t enabledCodeTypesMask) {
    LOGF("Running ONNX inference");
    
    // Create input tensor
    auto inputTensorOrt = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(),
        modelInfo.inputShape.data(), modelInfo.inputShape.size()
    );
    
    // Run inference
    const char* inputNames[] = {modelInfo.inputName.c_str()};
    const char* outputNames[] = {modelInfo.outputName.c_str()};
    
    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensorOrt, 1, outputNames, 1);
    
    // Extract output data
    auto& outputTensor = outputTensors[0];
    float* outputData = outputTensor.GetTensorMutableData<float>();
    auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    
    LOGF("ONNX inference completed, output shape: [%ld, %ld, %ld]", 
         (long)outputShape[0], (long)outputShape[1], (long)outputShape[2]);
    
    // Copy to vector for processing
    size_t outputSize = 1;
    for (auto dim : outputShape) outputSize *= dim;
    std::vector<float> output(outputData, outputData + outputSize);
    
    return findBestDetection(output, enabledCodeTypesMask);
}

DetectionResult OnnxProcessor::findBestDetection(const std::vector<float>& modelOutput, uint8_t enabledCodeTypesMask) {
    // Validate output format [1, 9, 8400]
    const size_t expectedFeatures = 9;
    const size_t expectedAnchors = 8400;
    
    if (modelOutput.size() != expectedFeatures * expectedAnchors) {
        LOGF("ERROR: Unexpected output size %zu, expected %zu", modelOutput.size(), expectedFeatures * expectedAnchors);
        return {};
    }
    
    LOGF("Processing %zu anchors for best detection with enabled types mask: 0x%02X", expectedAnchors, enabledCodeTypesMask);
    
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto getVal = [&](size_t anchorIdx, size_t featureIdx) -> float {
        return modelOutput[featureIdx * expectedAnchors + anchorIdx];
    };
    
    // Find best anchor among enabled code detection types only
    size_t bestAnchor = 0;
    float bestConfidence = 0.0f;
    int bestClass = -1;
    
    for (size_t a = 0; a < expectedAnchors; a++) {
        // Find best class for this anchor among enabled types only
        float maxClassProb = 0.0f;
        int classIdx = -1;
        for (int c = 0; c < CODE_DETECTION_CLASS_COUNT; c++) {
            // Check if this class is enabled using bitmask
            uint8_t classMask = 1 << c;
            if (!(enabledCodeTypesMask & classMask)) {
                continue; // Skip disabled classes
            }
            
            float classProb = sigmoid(getVal(a, 4 + c));
            if (classProb > maxClassProb) {
                maxClassProb = classProb;
                classIdx = c;
            }
        }
        
        if (maxClassProb > bestConfidence && maxClassProb > 0.55f) {
            bestConfidence = maxClassProb;
            bestAnchor = a;
            bestClass = classIdx;
        }
    }
    
    // Create result
    DetectionResult result;
    if (bestConfidence > 0.0f) {
        result.confidence = bestConfidence;
        result.centerX = getVal(bestAnchor, 0) / 640.0f;    // Normalize to [0,1]
        result.centerY = getVal(bestAnchor, 1) / 640.0f;    // Normalize to [0,1]
        result.width = getVal(bestAnchor, 2) / 640.0f;      // Normalize to [0,1]
        result.height = getVal(bestAnchor, 3) / 640.0f;     // Normalize to [0,1]
        result.classIndex = bestClass;
        
        CodeDetectionType detectionType = indexToCodeDetectionType(bestClass);
        LOGF("Best detection: class=%d (%s), conf=%.3f, coords=(%.3f,%.3f) size=%.3fx%.3f", 
             result.classIndex, getCodeDetectionClassName(detectionType), result.confidence, 
             result.centerX, result.centerY, result.width, result.height);
    } else {
        LOGF("No detection found above threshold for enabled types mask: 0x%02X", enabledCodeTypesMask);
    }
    
    return result;
}

} // namespace UniversalScanner