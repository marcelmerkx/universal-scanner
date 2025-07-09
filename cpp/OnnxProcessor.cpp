#include "OnnxProcessor.h"
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "UniversalScanner", fmt, ##__VA_ARGS__)

// Use getClassName from Universal.cpp to avoid duplicate symbol
extern const char* getClassName(int classIdx);

namespace UniversalScanner {

OnnxProcessor::OnnxProcessor() 
    : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), 
      modelLoaded(false), 
      modelSize_(320),  // Default to 320 for faster initialization
      yuvConverter(nullptr), 
      enableDebugImages(true),  // Default to true for debugging
      currentExecutionProvider(ExecutionProvider::CPU),
      assetManager_(nullptr),
      env_(nullptr) {
    LOGF("OnnxProcessor created");
    
    // Enable debug images by default in DEBUG builds, disable in release
    #ifdef DEBUG
        enableDebugImages = true;
        LOGF("Debug images enabled (DEBUG build)");
    #else
        enableDebugImages = true;  // Force enable for OCR debugging
        LOGF("Debug images force enabled for OCR debugging");
    #endif
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

// Load model from Android assets
std::vector<uint8_t> loadModelFromAssets(JNIEnv* env, jobject assetManager, const std::string& filename) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGF("Failed to get AAssetManager");
        return {};
    }
    
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        LOGF("Failed to open asset: %s", filename.c_str());
        return {};
    }
    
    off_t fileSize = AAsset_getLength(asset);
    std::vector<uint8_t> modelData(fileSize);
    
    int bytesRead = AAsset_read(asset, modelData.data(), fileSize);
    AAsset_close(asset);
    
    if (bytesRead != fileSize) {
        LOGF("Failed to read complete asset file");
        return {};
    }
    
    LOGF("Loaded ONNX model from assets: %s (%zu bytes)", filename.c_str(), modelData.size());
    return modelData;
}

bool OnnxProcessor::initializeModel() {
    if (modelLoaded) return true;
    
    try {
        // Update model info based on size
        modelInfo.inputShape = {1, 3, modelSize_, modelSize_}; // NCHW format
        // Output shape changes based on model size:
        // 320x320: 2100 anchors, 416x416: 3549 anchors, 640x640: 8400 anchors
        int numAnchors = (modelSize_ == 320) ? 2100 : (modelSize_ == 416) ? 3549 : 8400;
        modelInfo.outputShape = {1, 9, numAnchors}; // 9 features x anchors
        modelInfo.inputName = "images";
        modelInfo.outputName = "output0";
        
        // Build model filename based on size
        std::string modelFilename = "unified-detection-v7-" + std::to_string(modelSize_) + ".onnx";
        LOGF("Loading model: %s", modelFilename.c_str());
        
        // Try to load model from internal storage first
        std::string modelPath = "/data/data/com.cargosnap.universalscanner/files/" + modelFilename;
        auto modelData = loadModelFromFile(modelPath);
        if (modelData.empty() && env_ && assetManager_) {
            // Try loading from Android assets
            modelData = loadModelFromAssets(env_, assetManager_, modelFilename);
            if (modelData.empty()) {
                // No fallback - fail explicitly
                LOGF("❌ ERROR: Model %s not found for size %d", modelFilename.c_str(), modelSize_);
                LOGF("Tried paths:");
                LOGF("  - /data/data/com.cargosnap.universalscanner/files/%s", modelFilename.c_str());
                LOGF("  - Android assets: %s", modelFilename.c_str());
                return false;
            }
        } else if (modelData.empty()) {
            LOGF("❌ ERROR: Model %s not found and AssetManager not available", modelFilename.c_str());
            return false;
        }
        
        // Create ONNX Runtime environment
        ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "UniversalScanner");
        
        // Try to create session with hardware acceleration first, fallback to CPU if it fails
        bool sessionCreated = false;
        
        // First attempt: Try with hardware acceleration (NNAPI/CoreML)
        try {
            Ort::SessionOptions sessionOptions;
            
            // Use OnnxDelegateManager to select best execution provider - CPU only for investigation
            // Parameters: (sessionOptions, verbose, preferXNNPACK, disableNNAPI)
            currentExecutionProvider = OnnxDelegateManager::configure(sessionOptions, true, true, true);
            
            // Create session from memory buffer
            session = std::make_unique<Ort::Session>(*ortEnv, modelData.data(), modelData.size(), sessionOptions);
            sessionCreated = true;
            
            LOGF("✅ ONNX model loaded successfully with %s execution provider!", 
                 OnnxDelegateManager::getPerformanceDescription(currentExecutionProvider));
            LOGF("📊 Expected performance: %.1fx vs CPU baseline", 
                 OnnxDelegateManager::getPerformanceMultiplier(currentExecutionProvider));
                 
        } catch (const Ort::Exception& e) {
            LOGF("⚠️ Hardware acceleration failed (%s), falling back to CPU", e.what());
            
            // Second attempt: Fallback to CPU-only execution
            try {
                Ort::SessionOptions cpuSessionOptions;
                cpuSessionOptions.SetIntraOpNumThreads(1);
                // No execution provider configured = CPU default
                
                session = std::make_unique<Ort::Session>(*ortEnv, modelData.data(), modelData.size(), cpuSessionOptions);
                currentExecutionProvider = ExecutionProvider::CPU;
                sessionCreated = true;
                
                LOGF("✅ ONNX model loaded successfully with CPU fallback");
                LOGF("💡 Consider using a model optimized for mobile hardware acceleration");
                
            } catch (const Ort::Exception& cpuError) {
                LOGF("❌ CPU fallback also failed: %s", cpuError.what());
                return false;
            }
        }
        
        if (!sessionCreated) {
            LOGF("❌ Failed to create ONNX session with any execution provider");
            return false;
        }
        
        modelLoaded = true;
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
    LOGF("🎬 OnnxProcessor::processFrame called with %dx%d, %zu bytes", width, height, frameSize);
    PerformanceTimer totalTimer("TOTAL_FRAME_PROCESSING");
    
    try {
        // Store JNI environment for asset loading
        env_ = env;
        
        // Try to get AssetManager from the Android app if not already set
        if (!assetManager_) {
            // Look for an Android application context
            jclass activityThreadClass = env->FindClass("android/app/ActivityThread");
            if (activityThreadClass) {
                jmethodID currentApplicationMethod = env->GetStaticMethodID(activityThreadClass, "currentApplication", "()Landroid/app/Application;");
                if (currentApplicationMethod) {
                    jobject application = env->CallStaticObjectMethod(activityThreadClass, currentApplicationMethod);
                    if (application) {
                        jclass contextClass = env->GetObjectClass(application);
                        jmethodID getAssetsMethod = env->GetMethodID(contextClass, "getAssets", "()Landroid/content/res/AssetManager;");
                        if (getAssetsMethod) {
                            jobject localAssetManager = env->CallObjectMethod(application, getAssetsMethod);
                            if (localAssetManager) {
                                assetManager_ = env->NewGlobalRef(localAssetManager);
                                LOGF("✅ AssetManager obtained from application context");
                            }
                        }
                        env->DeleteLocalRef(contextClass);
                        env->DeleteLocalRef(application);
                    }
                }
                env->DeleteLocalRef(activityThreadClass);
            }
        }
        
        // Initialize components
        LOGF("🔧 Checking model initialization...");
        if (!modelLoaded && !initializeModel()) {
            LOGF("❌ Model initialization failed!");
            return {};
        }
        LOGF("✅ Model loaded successfully");
        
        LOGF("🔧 Initializing converters...");
        if (!initializeConverters(env, context)) {
            LOGF("❌ Converter initialization failed!");
            return {};
        }
        LOGF("✅ Converters initialized");
        
        // PHASE 1: Preprocessing (YUV resize + RGB conversion + rotation + padding)
        LOGF("🔄 Starting preprocessing phase...");
        PerformanceTimer preprocessTimer("PREPROCESSING");
        int processedWidth, processedHeight;
        auto rgbData = preprocessFrame(frameData, frameSize, width, height, &processedWidth, &processedHeight);
        if (rgbData.empty()) {
            LOGF("❌ Preprocessing failed - empty RGB data");
            return {};
        }
        LOGF("✅ Preprocessing complete: %dx%d", processedWidth, processedHeight);
        
        // Create ONNX tensor from preprocessed image
        LOGF("🧮 Creating tensor from RGB data...");
        auto inputTensor = createTensorFromRGB(rgbData, processedWidth, processedHeight);
        if (inputTensor.empty()) {
            LOGF("❌ Tensor creation failed");
            return {};
        }
        LOGF("✅ Tensor created: %zu elements", inputTensor.size());
        preprocessTimer.logElapsed("YUV→RGB→Rotate→Pad→Tensor");
        
        // PHASE 2: ONNX Inference
        LOGF("🧠 Starting ONNX inference phase...");
        PerformanceTimer inferenceTimer("INFERENCE");
        auto result = runInference(inputTensor, enabledCodeTypesMask);
        inferenceTimer.logElapsed("ONNX_MODEL_EXECUTION");
        LOGF("✅ ONNX inference complete");
        
        // Log total frame processing time with execution provider info
        totalTimer.logElapsed("COMPLETE_PIPELINE");
        LOGF("🚀 Frame processed using %s (%.2f ms total)", 
             getExecutionProviderName(), totalTimer.getElapsedMs());
        
        // Check if frame processing is too slow for real-time
        if (totalTimer.isSlowOperation()) {
            LOGF("⚠️ SLOW FRAME: %.2f ms (target: <33ms for 30 FPS) with %s", 
                 totalTimer.getElapsedMs(), getExecutionProviderName());
        }
        
        return result;
        
    } catch (const Ort::Exception& e) {
        LOGF("❌ ONNX inference error: %s", e.what());
        return {}; // NO FALLBACK MOCKING
    } catch (const std::exception& e) {
        LOGF("❌ Standard exception in processFrame: %s", e.what());
        return {};
    } catch (...) {
        LOGF("❌ Unknown exception in processFrame");
        return {};
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
        targetWidth = modelSize_;
        targetHeight = (height * modelSize_) / width;
    } else {
        targetHeight = modelSize_;
        targetWidth = (width * modelSize_) / height;
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
    
    // Apply white padding to make it square based on model size
    UniversalScanner::PaddingInfo padInfo;
    auto inputTensor = UniversalScanner::WhitePadding::applyPadding(rgbData, width, height, modelSize_, &padInfo);
    
    if (inputTensor.empty()) {
        LOGF("ERROR: White padding failed");
        return {};
    }
    
    if (enableDebugImages) {
        UniversalScanner::ImageDebugger::saveTensor("3_padded.jpg", inputTensor, modelSize_, modelSize_);
    }
    
    LOGF("Tensor created: %dx%d, first pixels: %.3f %.3f %.3f", modelSize_, modelSize_, 
         inputTensor[0], inputTensor[1], inputTensor[2]);
    
    return inputTensor;
}

DetectionResult OnnxProcessor::runInference(const std::vector<float>& inputTensor, uint8_t enabledCodeTypesMask) {
    LOGF("🚀 Running ONNX inference with %s provider", getExecutionProviderName());
    
    // PHASE 2A: ONNX Model Execution
    PerformanceTimer onnxTimer("ONNX_EXECUTION");
    
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
    
    onnxTimer.logElapsed("MODEL_FORWARD_PASS");
    
    LOGF("✅ ONNX inference completed, output shape: [%ld, %ld, %ld]", 
         (long)outputShape[0], (long)outputShape[1], (long)outputShape[2]);
    
    // DEBUG: Check some output values to see if model is producing reasonable results
    LOGF("🔬 First 10 output values: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f", 
         outputData[0], outputData[1], outputData[2], outputData[3], outputData[4],
         outputData[5], outputData[6], outputData[7], outputData[8], outputData[9]);
    
    // Check if all outputs are zeros (sign of a problem)
    bool allZeros = true;
    bool allNans = true;
    for (size_t i = 0; i < 100 && i < (outputShape[1] * outputShape[2]); i++) {
        if (outputData[i] != 0.0f) allZeros = false;
        if (!std::isnan(outputData[i])) allNans = false;
    }
    
    if (allZeros) {
        LOGF("⚠️ WARNING: All output values are zero - model may not be working correctly");
    }
    if (allNans) {
        LOGF("⚠️ WARNING: All output values are NaN - model has serious issues");
    }
    
    // PHASE 2B: Postprocessing (Detection Finding)
    PerformanceTimer postprocessTimer("POSTPROCESSING");
    
    // Copy to vector for processing
    size_t outputSize = 1;
    for (auto dim : outputShape) outputSize *= dim;
    std::vector<float> output(outputData, outputData + outputSize);
    
    auto result = findBestDetection(output, enabledCodeTypesMask);
    postprocessTimer.logElapsed("DETECTION_FILTERING");
    
    return result;
}

DetectionResult OnnxProcessor::findBestDetection(const std::vector<float>& modelOutput, uint8_t enabledCodeTypesMask) {
    DetectionResult result;
    result.hasDetection = false;
    result.confidence = 0.0f;
    
    // Validate output format based on model size
    const size_t expectedFeatures = 9;
    // Calculate expected anchors based on model size
    const size_t expectedAnchors = (modelSize_ == 320) ? 2100 : (modelSize_ == 416) ? 3549 : 8400;
    
    if (modelOutput.size() != expectedFeatures * expectedAnchors) {
        LOGF("❌ ERROR: Unexpected output size %zu, expected %zu (for %dx%d model)", 
             modelOutput.size(), expectedFeatures * expectedAnchors, modelSize_, modelSize_);
        return result;
    }
    
    LOGF("🔍 Processing %zu anchors for best detection with enabled types mask: 0x%02X", expectedAnchors, enabledCodeTypesMask);
    
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto getVal = [&](size_t anchorIdx, size_t featureIdx) -> float {
        return modelOutput[featureIdx * expectedAnchors + anchorIdx];
    };
    
    // DEBUG: Check first few anchor outputs
    LOGF("🔬 First anchor sample:");
    for (int f = 0; f < 9; f++) {
        LOGF("   Feature %d: %.6f", f, getVal(0, f));
    }
    
    // Find best anchor among enabled code detection types only
    size_t bestAnchor = 0;
    float bestConfidence = 0.0f;
    int bestClass = -1;
    int validAnchors = 0;
    int highConfidenceAnchors = 0;
    
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
        
        if (maxClassProb > 0.1f) validAnchors++;
        if (maxClassProb > 0.5f) highConfidenceAnchors++;
        
        if (maxClassProb > bestConfidence && maxClassProb > 0.55f) {
            bestConfidence = maxClassProb;
            bestAnchor = a;
            bestClass = classIdx;
            
            // Log promising detections
            if (maxClassProb > 0.6f) {
                LOGF("🎯 High confidence anchor %zu: class=%d, conf=%.3f, bbox=(%.3f,%.3f,%.3f,%.3f)", 
                     a, classIdx, maxClassProb, getVal(a, 0), getVal(a, 1), getVal(a, 2), getVal(a, 3));
            }
        }
    }
    
    LOGF("📊 Detection stats: %d valid (>0.1), %d high-conf (>0.5), %d passed threshold (>0.55)", 
         validAnchors, highConfidenceAnchors, (bestConfidence > 0.0f ? 1 : 0));
    
    // We already initialized result at the beginning with hasDetection = false
    if (bestConfidence > 0.0f) {
        result.hasDetection = true;
        result.confidence = bestConfidence;
        result.centerX = getVal(bestAnchor, 0) / static_cast<float>(modelSize_);    // Normalize to [0,1]
        result.centerY = getVal(bestAnchor, 1) / static_cast<float>(modelSize_);    // Normalize to [0,1]
        result.width = getVal(bestAnchor, 2) / static_cast<float>(modelSize_);      // Normalize to [0,1]
        result.height = getVal(bestAnchor, 3) / static_cast<float>(modelSize_);     // Normalize to [0,1]
        result.classIndex = bestClass;
        
        CodeDetectionType detectionType = indexToCodeDetectionType(bestClass);
        LOGF("✅ Best detection: class=%d (%s), conf=%.3f, coords=(%.3f,%.3f) size=%.3fx%.3f", 
             result.classIndex, getCodeDetectionClassName(detectionType), result.confidence, 
             result.centerX, result.centerY, result.width, result.height);
    } else {
        LOGF("❌ No detection found above threshold 0.55 for enabled types mask: 0x%02X", enabledCodeTypesMask);
        LOGF("💡 Best found was: conf=%.3f (anchor %zu)", bestConfidence, bestAnchor);
    }
    
    return result;
}

void OnnxProcessor::setModelSize(int size) {
    if (size != modelSize_) {
        LOGF("Model size changing from %d to %d, will reload model", modelSize_, size);
        modelSize_ = size;
        // Force model reload on next frame
        modelLoaded = false;
        // Reset session to free memory
        session.reset();
        ortEnv.reset();
    }
}

} // namespace UniversalScanner