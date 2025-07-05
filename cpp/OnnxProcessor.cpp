#include "OnnxProcessor.h"
#include <android/log.h>

#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "UniversalScanner", fmt, ##__VA_ARGS__)

// Use getClassName from Universal.cpp to avoid duplicate symbol
extern const char* getClassName(int classIdx);

namespace UniversalScanner {

OnnxProcessor::OnnxProcessor() 
    : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), modelLoaded(false), yuvConverter(nullptr) {
    LOGF("OnnxProcessor created");
    
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

std::vector<float> OnnxProcessor::processFrame(int width, int height, JNIEnv* env, jobject context, 
                                               const uint8_t* frameData, size_t frameSize) {
    // Initialize model if not already loaded
    if (!modelLoaded) {
        LOGF("Attempting to initialize ONNX model...");
        if (!initializeModel()) {
            LOGF("Failed to initialize ONNX model!");
            std::vector<float> empty;
            return empty; // Return empty - NO MOCKING
        }
    }
    
    // Initialize YUV converter and resizer if not already done
    if (!yuvConverter && env) {
        LOGF("Initializing platform-specific YUV converter...");
        yuvConverter = YuvConverter::create(env, context);
        if (!yuvConverter) {
            LOGF("Failed to create YUV converter");
        }
    }
    
    if (!yuvResizer && env) {
        LOGF("Initializing platform-specific YUV resizer...");
        yuvResizer = YuvResizer::create(env, context);
        if (!yuvResizer) {
            LOGF("Failed to create YUV resizer");
        }
    }
    
    if (!modelLoaded || !session) {
        LOGF("Model not loaded!");
        std::vector<float> empty;
        return empty; // Return empty - NO MOCKING
    }
    
    try {
        LOGF("Running REAL ONNX inference on %dx%d frame", width, height);
        
        // Extract real camera frame data - NO FALLBACKS, NO MOCK DATA
        if (!frameData || frameSize == 0) {
            LOGF("ERROR: No real frame data provided - cannot proceed without real camera data");
            std::vector<float> empty;
            return empty;
        }
        
        LOGF("Processing REAL camera frame data: %zu bytes", frameSize);
        
        // DEBUG: Save original YUV frame to compare with RGB conversion
        LOGF("DEBUG: Saving original YUV frame for comparison (%dx%d)", width, height);
        size_t ySize = width * height;
        size_t uvSize = ySize / 4;
        const uint8_t* yPlane = frameData;
        const uint8_t* uPlane = frameData + ySize;
        const uint8_t* vPlane = frameData + ySize + uvSize;
        UniversalScanner::ImageDebugger::saveYUV420("0_original_yuv.jpg", 
            yPlane, uPlane, vPlane, width, height, width, width / 2);
        
        // Step 1: Resize YUV for performance optimization
        LOGF("Step 1: Resizing YUV from %dx%d to optimize processing...", width, height);
        
        // Calculate target size - aim for ~640 pixels on longer dimension
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
            LOGF("Resizing YUV: %dx%d -> %dx%d (%.1fx reduction)", 
                 width, height, targetWidth, targetHeight, 
                 float(width * height) / float(targetWidth * targetHeight));
            
            resizedYuvData = yuvResizer->resizeYuv(frameData, frameSize, 
                                                  width, height, 
                                                  targetWidth, targetHeight);
            
            if (!resizedYuvData.empty()) {
                processWidth = targetWidth;
                processHeight = targetHeight;
                processFrameData = resizedYuvData.data();
                LOGF("YUV resize successful - processing %dx%d frame", processWidth, processHeight);
                
                // DEBUG: Save resized YUV frame
                size_t resizedYSize = processWidth * processHeight;
                size_t resizedUvSize = resizedYSize / 4;
                const uint8_t* resizedYPlane = processFrameData;
                const uint8_t* resizedUPlane = processFrameData + resizedYSize;
                const uint8_t* resizedVPlane = processFrameData + resizedYSize + resizedUvSize;
                UniversalScanner::ImageDebugger::saveYUV420("0b_resized_yuv.jpg", 
                    resizedYPlane, resizedUPlane, resizedVPlane, 
                    processWidth, processHeight, processWidth, processWidth / 2);
            } else {
                LOGF("YUV resize failed - using original dimensions");
            }
        } else {
            LOGF("Skipping YUV resize - using original %dx%d", width, height);
        }
        
        // Step 2: Convert YUV to RGB using platform-specific converter
        LOGF("Step 2: Converting YUV to RGB using platform-specific converter...");
        
        std::vector<uint8_t> rgbData;
        if (yuvConverter) {
            LOGF("Using platform-specific YUV converter (Android/iOS optimized)");
            size_t processFrameSize = resizedYuvData.empty() ? frameSize : resizedYuvData.size();
            rgbData = yuvConverter->convertYuvToRgb(processFrameData, processFrameSize, 
                                                   processWidth, processHeight);
        } else {
            LOGF("ERROR: Platform-specific YUV converter not available - cannot proceed");
            LOGF("Legacy YUV conversion has been deprecated due to stride/format issues");
            std::vector<float> empty;
            return empty;
        }
        
        if (rgbData.empty()) {
            LOGF("ERROR: YUV to RGB conversion failed");
            std::vector<float> empty;
            return empty;
        }
        
        // DEBUG: Save RGB data after YUV conversion
        LOGF("DEBUG: Saving RGB data after YUV conversion (%dx%d)", processWidth, processHeight);
        UniversalScanner::ImageDebugger::saveRGB("1_rgb_converted.jpg", rgbData, processWidth, processHeight);
        
        // Step 3: Apply 90° CW rotation if needed
        size_t frameWidth = processWidth;
        size_t frameHeight = processHeight;
        
        // Apply 90° CW rotation to fix orientation from emulator  
        LOGF("Step 3: Applying 90° CW rotation to fix orientation (%zux%zu)", frameWidth, frameHeight);
        rgbData = UniversalScanner::ImageRotation::rotate90CW(rgbData, frameWidth, frameHeight);
        // Dimensions are swapped for 90° rotation
        std::swap(frameWidth, frameHeight);
        
        // DEBUG: Save RGB data after rotation
        LOGF("DEBUG: Saving RGB data after 90° CW rotation (%zux%zu)", frameWidth, frameHeight);
        UniversalScanner::ImageDebugger::saveRGB("2_rotated.jpg", rgbData, frameWidth, frameHeight);
        
        // Step 4: Apply white padding to make it square 640x640
        LOGF("Step 4: Applying white padding to %zux%zu -> 640x640", frameWidth, frameHeight);
        UniversalScanner::PaddingInfo padInfo;
        std::vector<float> inputTensor = UniversalScanner::WhitePadding::applyPadding(
            rgbData, frameWidth, frameHeight, 640, &padInfo
        );
        
        if (inputTensor.empty()) {
            LOGF("ERROR: White padding failed");
            std::vector<float> empty;
            return empty;
        }
        
        // DEBUG: Save padded tensor data
        LOGF("DEBUG: Saving padded tensor data (640x640)");
        UniversalScanner::ImageDebugger::saveTensor("3_padded.jpg", inputTensor, 640, 640);
        
        LOGF("Real frame preprocessing completed - padded to 640x640, first few pixels: %.3f %.3f %.3f", 
             inputTensor[0], inputTensor[1], inputTensor[2]);
        
        // Create input tensor
        auto inputTensorOrt = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensor.data(),
            inputTensor.size(),
            modelInfo.inputShape.data(),
            modelInfo.inputShape.size()
        );
        
        // Run inference
        const char* inputNames[] = {modelInfo.inputName.c_str()};
        const char* outputNames[] = {modelInfo.outputName.c_str()};
        
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensorOrt,
            1,
            outputNames,
            1
        );
        
        // Get output data and structure it properly like OnnxPlugin.cpp
        auto& outputTensor = outputTensors[0];
        float* outputData = outputTensor.GetTensorMutableData<float>();
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
        
        LOGF("ONNX inference completed, output shape: [%ld, %ld, %ld]", 
             (long)outputShape[0], (long)outputShape[1], (long)outputShape[2]);
        
        // Copy raw output to vector for processing
        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }
        std::vector<float> output(outputData, outputData + outputSize);
        
        // Following OnnxPlugin.cpp approach: YOLOv8n model outputs [1, 9, 8400]
        // Create proper nested structure for processing like ONNX-RN format
        const size_t batch = outputShape[0];      // 1
        const size_t features = outputShape[1];   // 9 
        const size_t anchors = outputShape[2];    // 8400
        
        LOGF("Processing ONNX output: batch=%zu, features=%zu, anchors=%zu", batch, features, anchors);
        
        // Helper function for sigmoid activation
        auto sigmoid = [](float x) {
            return 1.0f / (1.0f + std::exp(-x));
        };
        
        // Parse using proper indexing: [batch, features, anchors] layout
        float bestConfidence = 0.0f;
        float bestX = 0.0f, bestY = 0.0f, bestW = 0.0f, bestH = 0.0f;
        int bestClass = -1;
        
        // Determine tensor format - could be [1, 9, 8400] or [1, 8400, 9]
        bool isFeaturesMajor = true; // [1, 9, 8400] format
        
        // Sample a few anchors to determine format
        float sample_obj_fm = output[4 * anchors + 100]; // features-major format
        float sample_obj_am = output[100 * features + 4]; // anchors-major format
        
        // YOLO objectness logits should be in range ~[-10, 10], not 20+
        if (std::abs(sample_obj_fm) > 15.0f && std::abs(sample_obj_am) < 15.0f) {
            isFeaturesMajor = false;
            LOGF("Detected tensor format: [1, 8400, 9] (anchors-major)");
        } else {
            LOGF("Detected tensor format: [1, 9, 8400] (features-major)");
        }
        
        // Helper lambda for adaptive indexing
        auto getVal = [&](size_t anchorIdx, size_t featureIdx) -> float {
            if (isFeaturesMajor) {
                // [1, 9, 8400] format: feature * anchors + anchor
                return output[featureIdx * anchors + anchorIdx];
            } else {
                // [1, 8400, 9] format: anchor * features + feature
                return output[anchorIdx * features + featureIdx];
            }
        };
        
        // Process all anchors with adaptive indexing
        for (size_t a = 0; a < anchors; a++) {
            // Get raw bbox coordinates (features 0-3) - these are in YOLO format
            float x_center = getVal(a, 0);
            float y_center = getVal(a, 1);
            float width_f   = getVal(a, 2);
            float height_f  = getVal(a, 3);
            
            // This model has NO objectness score! Just 5 class scores directly
            // Features 4-8: 5 class probabilities 
            float maxClassProb = 0.0f;
            int classIdx = -1;
            for (int c = 0; c < 5; c++) {  // 5 classes: features 4-8
                float classProb_raw = getVal(a, 4 + c);
                float classProb = sigmoid(classProb_raw);
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classIdx = c;
                }
            }
            
            // No objectness - just use class probability directly
            float confidence = maxClassProb;
            
            if (confidence > bestConfidence && confidence > 0.55f) {
                bestConfidence = confidence;
                bestX = x_center;
                bestY = y_center;
                bestW = width_f;
                bestH = height_f;
                bestClass = classIdx;
            }
        }
        
        LOGF("Best detection: class=%d, conf=%.3f", bestClass, bestConfidence);
        
        std::vector<float> results;
        if (bestConfidence > 0.0f) {
            // Return raw ONNX coordinates normalized to [0,1] for 640x640 space
            results.push_back(bestConfidence);
            results.push_back(bestX / 640.0f);         // x in [0,1]
            results.push_back(bestY / 640.0f);         // y in [0,1] 
            results.push_back(bestW / 640.0f);         // w in [0,1]
            results.push_back(bestH / 640.0f);         // h in [0,1]
            results.push_back(bestClass);              // class index
            
            LOGF("Returning normalized coords: x=%.3f, y=%.3f, w=%.3f, h=%.3f", 
                 bestX/640.0f, bestY/640.0f, bestW/640.0f, bestH/640.0f);
        }
        return results;
        
    } catch (const Ort::Exception& e) {
        LOGF("ONNX inference error: %s", e.what());
        std::vector<float> empty;
        return empty; // NO FALLBACK MOCKING
    }
}

} // namespace UniversalScanner