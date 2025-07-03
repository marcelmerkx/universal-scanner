#include <jni.h>
#include <fbjni/fbjni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>
#include <cmath>

// Include ONNX Runtime
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

// Include existing preprocessing modules
#include "preprocessing/FrameConverter.h"
#include "preprocessing/ImageRotation.h"
#include "preprocessing/WhitePadding.h"

#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "UniversalNative", fmt, ##__VA_ARGS__)

using namespace facebook::jni;

namespace universal {

// Real ONNX session wrapper for native method calls
class NativeOnnxSession {
private:
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::Env> ortEnv;
    Ort::MemoryInfo memoryInfo;
    bool modelLoaded;
    
    // Model info for unified-detection-v7.onnx
    struct {
        std::vector<int64_t> inputShape = {1, 3, 640, 640}; // NCHW format
        std::vector<int64_t> outputShape = {1, 9, 8400}; // 9 features x 8400 anchors
        std::string inputName = "images";
        std::string outputName = "output0";
    } modelInfo;
    
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
    
public:
    NativeOnnxSession() : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), modelLoaded(false) {
        LOGF("NativeOnnxSession created");
    }
    
    bool initializeModel() {
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
    
    // Run real ONNX inference with existing preprocessing pipeline
    std::vector<float> processFrame(int width, int height, JNIEnv* env, jobject context, const uint8_t* frameData, size_t frameSize) {
        // Initialize model if not already loaded
        if (!modelLoaded) {
            LOGF("Attempting to initialize ONNX model...");
            if (!initializeModel()) {
                LOGF("Failed to initialize ONNX model!");
                std::vector<float> empty;
                return empty; // Return empty - NO MOCKING
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
            
            // Step 1: Convert YUV byte array to RGB directly
            LOGF("Step 1: Converting YUV to RGB from real camera data...");
            
            // Calculate YUV plane sizes for YUV_420_888 format
            size_t ySize = width * height;
            size_t uvSize = ySize / 4;
            size_t expectedSize = ySize + uvSize * 2; // Y + U + V
            
            if (frameSize < expectedSize) {
                LOGF("ERROR: Frame size %zu is smaller than expected %zu for %dx%d YUV_420_888", 
                     frameSize, expectedSize, width, height);
                std::vector<float> empty;
                return empty;
            }
            
            // YUV_420_888 layout: Y plane, then U/V planes
            const uint8_t* yPlane = frameData;
            const uint8_t* uPlane = frameData + ySize;
            const uint8_t* vPlane = frameData + ySize + uvSize;
            
            // Convert YUV planes to RGB
            std::vector<uint8_t> rgbData = UniversalScanner::FrameConverter::convertI420toRGB(
                yPlane, uPlane, vPlane, width, height, width, width / 2
            );
            if (rgbData.empty()) {
                LOGF("ERROR: YUV to RGB conversion failed");
                std::vector<float> empty;
                return empty;
            }
            
            // Step 2: Apply 90° CCW rotation if needed (640x480 -> 480x640)
            size_t frameWidth = width;
            size_t frameHeight = height;
            
            if (frameWidth > frameHeight) { // 640x480 needs rotation
                LOGF("Step 2: Applying 90° CCW rotation (%zux%zu -> %zux%zu)", frameWidth, frameHeight, frameHeight, frameWidth);
                rgbData = UniversalScanner::ImageRotation::rotate90CCW(rgbData, frameWidth, frameHeight);
                std::swap(frameWidth, frameHeight); // Now 480x640
            } else {
                LOGF("Step 2: No rotation needed for %zux%zu", frameWidth, frameHeight);
            }
            
            // Step 3: Apply white padding to make it square 640x640
            LOGF("Step 3: Applying white padding to %zux%zu -> 640x640", frameWidth, frameHeight);
            UniversalScanner::PaddingInfo padInfo;
            std::vector<float> inputTensor = UniversalScanner::WhitePadding::applyPadding(
                rgbData, frameWidth, frameHeight, 640, &padInfo
            );
            
            if (inputTensor.empty()) {
                LOGF("ERROR: White padding failed");
                std::vector<float> empty;
                return empty;
            }
            
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
            
            LOGF("First few raw values from tensor: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f", 
                 output[0], output[1], output[2], output[3], output[4], 
                 output[5], output[6], output[7], output[8]);
            
            // Determine tensor format - could be [1, 9, 8400] or [1, 8400, 9]
            // Check by looking at reasonable value ranges for objectness scores
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
            
            // Debug: Check tensor layout with adaptive indexing
            LOGF("Checking tensor layout - testing with adaptive indexing:");
            
            // Check the actual raw indices to understand the layout
            LOGF("Raw tensor values at key positions (total size: %zu):", output.size());
            if (33600 < output.size()) {
                LOGF("  output[33600] (feat 4, anchor 0 if [1,9,8400]): %.3f", output[33600]);
            } else {
                LOGF("  output[33600] would be out of bounds");
            }
            LOGF("  output[4] (feat 4, anchor 0 if [1,8400,9]): %.3f", output[4]);
            
            // Test with getVal
            LOGF("Using getVal adaptive indexing:");
            LOGF("  anchor[0] x=%.3f, y=%.3f, w=%.3f, h=%.3f", 
                 getVal(0, 0), getVal(0, 1), getVal(0, 2), getVal(0, 3));
            LOGF("  anchor[0] objectness_raw=%.3f (sigmoid=%.3f)", 
                 getVal(0, 4), sigmoid(getVal(0, 4)));
            LOGF("  anchor[0] class scores: [%.3f, %.3f, %.3f, %.3f]",
                 sigmoid(getVal(0, 5)), sigmoid(getVal(0, 6)), 
                 sigmoid(getVal(0, 7)), sigmoid(getVal(0, 8)));
            
            // Deep analysis of ONNX output structure
            LOGF("=== DEEP ONNX OUTPUT ANALYSIS ===");
            
            // Check different interpretations of the 9 features
            LOGF("Testing different feature interpretations for anchor 952 (where you found good values before):");
            size_t testAnchor = 952;
            for (size_t f = 0; f < features; f++) {
                float val = getVal(testAnchor, f);
                LOGF("  Feature[%zu]: raw=%.3f, sigmoid=%.3f", f, val, sigmoid(val));
            }
            
            // Find anchors with non-zero values in different positions
            LOGF("Searching for anchors with significant values:");
            int foundCount = 0;
            for (size_t a = 0; a < anchors && foundCount < 10; a++) {
                bool hasSignificant = false;
                for (size_t f = 4; f < features; f++) { // Check from feature 4 onwards
                    float val = getVal(a, f);
                    if (std::abs(val) > 0.1f) { // Non-zero threshold
                        hasSignificant = true;
                        break;
                    }
                }
                if (hasSignificant) {
                    LOGF("Anchor %zu has non-zero values:", a);
                    for (size_t f = 0; f < features; f++) {
                        float val = getVal(a, f);
                        LOGF("  [%zu]=%.3f", f, val);
                    }
                    foundCount++;
                }
            }
            
            // Dump entire ONNX output to file for analysis
            std::string outputFile = "/storage/emulated/0/Download/onnx_output.txt";
            FILE* file = fopen(outputFile.c_str(), "w");
            if (file) {
                fprintf(file, "ONNX Output Analysis\n");
                fprintf(file, "Output shape: [%ld, %ld, %ld]\n", 
                       (long)outputShape[0], (long)outputShape[1], (long)outputShape[2]);
                fprintf(file, "Total elements: %ld\n", 
                       (long)(outputShape[0] * outputShape[1] * outputShape[2]));
                
                // Analyze value distribution
                fprintf(file, "\n--- VALUE DISTRIBUTION ANALYSIS ---\n");
                int zeroCount = 0, nearZeroCount = 0, smallCount = 0, largeCount = 0;
                float minVal = 1e9, maxVal = -1e9;
                for (size_t i = 0; i < output.size(); i++) {
                    float val = output[i];
                    if (val == 0.0f) zeroCount++;
                    else if (std::abs(val) < 0.01f) nearZeroCount++;
                    else if (std::abs(val) < 1.0f) smallCount++;
                    else largeCount++;
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
                fprintf(file, "Zeros: %d, Near-zeros: %d, Small: %d, Large: %d\n", 
                        zeroCount, nearZeroCount, smallCount, largeCount);
                fprintf(file, "Min: %.6f, Max: %.6f\n", minVal, maxVal);
                
                fprintf(file, "\n--- STRUCTURED BY ANCHOR (first 100 anchors) ---\n");
                fprintf(file, "Format: %s\n", isFeaturesMajor ? "[1, 9, 8400]" : "[1, 8400, 9]");
                for (size_t anchor = 0; anchor < std::min(size_t(100), anchors); anchor++) {
                    fprintf(file, "Anchor %zu: ", anchor);
                    for (size_t feat = 0; feat < features; feat++) {
                        float rawVal = getVal(anchor, feat);
                        fprintf(file, "%.3f ", rawVal);
                    }
                    // Try different confidence calculations
                    float obj = sigmoid(getVal(anchor, 4));
                    float cls0 = sigmoid(getVal(anchor, 5));
                    float cls1 = sigmoid(getVal(anchor, 6));
                    float conf1 = obj * cls0;
                    float conf2 = obj * cls1;
                    float maxConf = std::max(conf1, conf2);
                    fprintf(file, " -> obj=%.3f, cls0=%.3f, cls1=%.3f, maxConf=%.3f\n", 
                            obj, cls0, cls1, maxConf);
                }
                
                fclose(file);
                LOGF("ONNX output dumped to: %s", outputFile.c_str());
            } else {
                LOGF("Failed to create output file: %s", outputFile.c_str());
            }
            
            LOGF("About to start detection search using OnnxPlugin.cpp approach...");
            
            // Process anchors but focus on regions with realistic YOLO logit values
            // From earlier analysis: anchors ~952 (element 8570) have proper negative values
            // The tensor has structured regions - we need to find the actual detection data
            
            // Find best detection - but let's analyze the tensor structure first
            int candidatesAbove10 = 0, candidatesAbove25 = 0, candidatesAbove50 = 0, candidatesAbove70 = 0;
            
            // Process all anchors with adaptive indexing
            for (size_t a = 0; a < anchors; a++) {
                // Get raw bbox coordinates (features 0-3) - these are in YOLO format
                float x_center = getVal(a, 0);
                float y_center = getVal(a, 1);
                float width    = getVal(a, 2);
                float height   = getVal(a, 3);
                
                // Get objectness score (feature 4) - this is a raw logit, needs sigmoid
                float objectness_raw = getVal(a, 4);
                float objectness = sigmoid(objectness_raw);
                
                // Get class probabilities (features 5-8 for 4 classes, NOT 5!)
                // Model has 9 features total: 4 bbox + 1 objectness + 4 classes
                float maxClassProb = 0.0f;
                int classIdx = -1;
                for (int c = 0; c < 4; c++) {
                    float classProb_raw = getVal(a, 5 + c);
                    float classProb = sigmoid(classProb_raw);
                    if (classProb > maxClassProb) {
                        maxClassProb = classProb;
                        classIdx = c;
                    }
                }
                
                // Try different confidence calculations
                // Some YOLO variants use just class probability without objectness
                float confMethod1 = objectness * maxClassProb;  // Traditional YOLO
                float confMethod2 = maxClassProb;               // Just class probability
                float confMethod3 = std::max(objectness, maxClassProb); // Max of both
                
                // Use the method that gives reasonable confidence scores
                float confidence = confMethod2;  // Try just class probability
                
                // Debug high-confidence detections
                if (confidence > 0.5f && a >= 6900 && a <= 7000) {
                    LOGF("HIGH CONF anchor %zu: obj=%.3f, maxClass=%.3f, method1=%.3f, method2=%.3f, bbox=[%.1f,%.1f,%.1f,%.1f]",
                         a, objectness, maxClassProb, confMethod1, confMethod2, x_center, y_center, width, height);
                    
                    // Show all class scores for best anchors
                    if (a >= 6930 && a <= 6940) {
                        LOGF("  All classes for anchor %zu:", a);
                        for (int c = 0; c < 4; c++) {
                            float raw = getVal(a, 5 + c);
                            float prob = sigmoid(raw);
                            LOGF("    Class[%d]: raw=%.3f, sigmoid=%.3f", c, raw, prob);
                        }
                    }
                }
                
                // Count candidates at different thresholds
                if (confidence > 0.10f) candidatesAbove10++;
                if (confidence > 0.25f) candidatesAbove25++;
                if (confidence > 0.50f) candidatesAbove50++;
                if (confidence > 0.70f) candidatesAbove70++;
                
                // Log detections above 70% (the target threshold) 
                if (confidence > 0.70f) {
                    LOGF("TARGET CONFIDENCE %zu: class=%d, obj=%.3f, classProb=%.3f, conf=%.3f, bbox=[%.1f,%.1f,%.1f,%.1f]", 
                         a, classIdx, objectness, maxClassProb, confidence, 
                         x_center, y_center, width, height);
                }
                
                if (confidence > bestConfidence && confidence > 0.25f) {
                    bestConfidence = confidence;
                    bestX = x_center;
                    bestY = y_center;
                    bestW = width;
                    bestH = height;
                    bestClass = classIdx;
                }
            }
            
            LOGF("Detection stats: %d candidates >10%%, %d >25%%, %d >50%%, %d >70%%. Best: %.3f%%", 
                 candidatesAbove10, candidatesAbove25, candidatesAbove50, candidatesAbove70, bestConfidence * 100);
            
            // Show all detections above 40% using adaptive indexing
            for (size_t a = 0; a < anchors && a < 100; a++) { // Limit to first 100 to avoid spam
                // Get objectness and class scores using adaptive indexing
                float objectness_raw = getVal(a, 4);
                float objectness = sigmoid(objectness_raw);
                
                float maxClassProb = 0.0f;
                int classIdx = -1;
                for (int c = 0; c < 4; c++) {
                    float classProb_raw = getVal(a, 5 + c);
                    float classProb = sigmoid(classProb_raw);
                    if (classProb > maxClassProb) {
                        maxClassProb = classProb;
                        classIdx = c;
                    }
                }
                // Use same confidence calculation as main loop
                float confidence = maxClassProb;  // Just class probability
                
                if (confidence > 0.40f) { // Show all >40% detections
                    float x = getVal(a, 0);
                    float y = getVal(a, 1);
                    float w = getVal(a, 2);
                    float h = getVal(a, 3);
                    
                    LOGF("DETECTION[%zu]: {\"type\":\"license_plate\", \"confidence\":%.3f, \"x\":%d, \"y\":%d, \"width\":%d, \"height\":%d, \"model\":\"unified-detection-v7.onnx\"}", 
                         a, confidence, (int)(x * width / 640.0f), (int)(y * height / 640.0f), 
                         (int)(w * width / 640.0f), (int)(h * height / 640.0f));
                }
            }
            
            LOGF("Best detection: class=%d, conf=%.3f, x=%.3f, y=%.3f, w=%.3f, h=%.3f", 
                 bestClass, bestConfidence, bestX, bestY, bestW, bestH);
            
            std::vector<float> results;
            if (bestConfidence > 0.0f) {
                // Normalize coordinates from pixel space to [0,1]
                results.push_back(bestConfidence);
                results.push_back(bestX / 640.0f);
                results.push_back(bestY / 640.0f);
                results.push_back(bestW / 640.0f);
                results.push_back(bestH / 640.0f);
            }
            return results;
            
        } catch (const Ort::Exception& e) {
            LOGF("ONNX inference error: %s", e.what());
            std::vector<float> empty;
            return empty; // NO FALLBACK MOCKING
        }
    }
};

// Global session instance
static std::unique_ptr<NativeOnnxSession> g_nativeSession = nullptr;

class UniversalNativeModule : public HybridClass<UniversalNativeModule> {
public:
    static constexpr auto kJavaDescriptor = "Lcom/universal/UniversalNativeModule;";
    
    static local_ref<jhybriddata> initHybrid(alias_ref<jclass>) {
        return makeCxxInstance();
    }
    
    static void registerNatives() {
        registerHybrid({
            makeNativeMethod("initHybrid", UniversalNativeModule::initHybrid),
            makeNativeMethod("nativeProcessFrame", UniversalNativeModule::nativeProcessFrame),
            makeNativeMethod("nativeProcessFrameWithData", UniversalNativeModule::nativeProcessFrameWithData),
        });
    }
    
    // Native method that Java can call for real inference
    local_ref<jstring> nativeProcessFrame(int width, int height) {
        LOGF("nativeProcessFrame called with %dx%d", width, height);
        
        try {
            // Initialize session if needed
            if (!g_nativeSession) {
                g_nativeSession = std::make_unique<NativeOnnxSession>();
            }
            
            // Get JNI environment
            JNIEnv* jniEnv = facebook::jni::Environment::current();
            
            // For now, pass nullptr for context - we'll load the model differently
            jobject contextObj = nullptr;
            
            // Extract real frame data from VisionCamera and process it
            LOGF("Extracting real frame data from VisionCamera...");
            
            // TODO: We need to extract actual frame data from VisionCamera
            // For now, return empty until we get the Frame object passed properly
            LOGF("Frame data extraction not implemented - need VisionCamera Frame object");
            return make_jstring("{\"detections\":[]}");
            
        } catch (const std::exception& e) {
            LOGF("Error in nativeProcessFrameWithContext: %s", e.what());
            return make_jstring("{\"error\":\"" + std::string(e.what()) + "\"}");
        }
    }
    
    // Native method that processes real frame data from VisionCamera
    local_ref<jstring> nativeProcessFrameWithData(int width, int height, alias_ref<jbyteArray> frameData) {
        LOGF("nativeProcessFrameWithData called with %dx%d, frame data size: %zu", width, height, frameData->size());
        
        try {
            // Initialize session if needed
            if (!g_nativeSession) {
                g_nativeSession = std::make_unique<NativeOnnxSession>();
            }
            
            // Get JNI environment
            JNIEnv* jniEnv = facebook::jni::Environment::current();
            
            // Extract frame data from Java byte array
            jsize dataSize = frameData->size();
            auto frameDataRegion = frameData->getRegion(0, dataSize);
            const uint8_t* frameBytes = reinterpret_cast<const uint8_t*>(frameDataRegion.get());
            
            LOGF("Processing REAL VisionCamera frame: %dx%d, %zu bytes", width, height, dataSize);
            
            // Call the real ONNX processing with actual frame data
            std::vector<float> results = g_nativeSession->processFrame(
                width, height, jniEnv, nullptr, frameBytes, dataSize
            );
            
            // Format results as JSON
            if (results.empty()) {
                return make_jstring("{\"detections\":[]}");
            }
            
            // results format: [confidence, x, y, w, h]
            std::string json = "{\"detections\":[{";
            json += "\"type\":\"license_plate\",";
            json += "\"confidence\":" + std::to_string(results[0]) + ",";
            json += "\"x\":" + std::to_string((int)(results[1] * width)) + ",";
            json += "\"y\":" + std::to_string((int)(results[2] * height)) + ",";
            json += "\"width\":" + std::to_string((int)(results[3] * width)) + ",";
            json += "\"height\":" + std::to_string((int)(results[4] * height)) + ",";
            json += "\"model\":\"unified-detection-v7.onnx\"";
            json += "}]}";
            
            return make_jstring(json);
            
        } catch (const std::exception& e) {
            LOGF("Error in nativeProcessFrameWithData: %s", e.what());
            return make_jstring("{\"error\":\"" + std::string(e.what()) + "\"}");
        }
    }
};

} // namespace universal

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    return facebook::jni::initialize(vm, [] {
        universal::UniversalNativeModule::registerNatives();
    });
}