#include "OnnxProcessorV2.h"
#include "preprocessing/CropExtractor.h"
#include "preprocessing/AdaptiveLetterbox.h"
#include "preprocessing/FrameConverter.h"
#include "preprocessing/ImageDebugger.h"
#include "ocr/ContainerOCRProcessor.h"
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <fstream>
#include <algorithm>
#include <cstring>

#define LOG_TAG "OnnxProcessorV2"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace UniversalScanner {

OnnxProcessorV2::OnnxProcessorV2() : OnnxProcessor() {
    LOGD("OnnxProcessorV2 created with two-stage pipeline support");
}

bool OnnxProcessorV2::initializeOCR(JNIEnv* env, jobject assetManager) {
    LOGD("🔧 Starting OCR initialization...");
    try {
        // Check if OCR model file exists
        std::string ocrModelPath = "/data/data/com.cargosnap.universalscanner/files/best-OCR-Colab-22-06-25.onnx";
        
        // Check if file exists
        std::ifstream file(ocrModelPath);
        if (!file.good()) {
            LOGE("❌ OCR model file not found at: %s", ocrModelPath.c_str());
            return false;
        }
        file.close();
        
        LOGD("🔧 OCR model file found at: %s", ocrModelPath.c_str());
        LOGD("🔧 Initializing OCR engine with model: %s", ocrModelPath.c_str());
        
        // Initialize OCR engine with the actual OCR model
        ocrEngine_ = std::make_unique<universalscanner::YoloOCREngine>(ocrModelPath);
        LOGD("✅ OCR engine initialized successfully");
        
        // Set up the processor registry for all supported code types
        processors_["code_container_v"] = universalscanner::ContainerOCRProcessor::processContainerCode;
        processors_["code_container_h"] = universalscanner::ContainerOCRProcessor::processContainerCode;
        processors_["code_seal"] = universalscanner::ContainerOCRProcessor::processContainerCode;
        processors_["code_qr_barcode"] = universalscanner::ContainerOCRProcessor::processContainerCode;
        processors_["code_license_plate"] = universalscanner::ContainerOCRProcessor::processContainerCode;
        
        LOGD("✅ OCR processor registry initialized with %zu processors", processors_.size());
        LOGD("✅ OCR initialization complete!");
        return true;
        
    } catch (const std::exception& e) {
        LOGE("❌ Failed to initialize OCR: %s", e.what());
        return false;
    }
}

std::vector<universalscanner::ScanResult> OnnxProcessorV2::processFrameWithOCR(
    int width, int height,
    JNIEnv* env, jobject context,
    const uint8_t* frameData, size_t frameSize,
    uint8_t enabledCodeTypesMask
) {
    // Stage 1: Detection using base processor
    auto detectionResult = processFrame(
        width, height, env, context,
        frameData, frameSize, enabledCodeTypesMask
    );
    
    if (!detectionResult.isValid()) {
        return std::vector<universalscanner::ScanResult>();  // No detections
    }
    
    // Use the new method that doesn't re-run detection
    return processOCRWithDetection(detectionResult, width, height, frameData, frameSize);
}

std::vector<universalscanner::ScanResult> OnnxProcessorV2::processOCRWithDetection(
    const DetectionResult& detectionResult,
    int width, int height,
    const uint8_t* frameData, size_t frameSize
) {
    std::vector<universalscanner::ScanResult> results;
    
    if (detectionResult.confidence < 0.5f) {
        return results;
    }
    
    // ====== COORDINATE CONVERSION (as per detection_to_ocr_pipeline_fix.md) ======
    // Detection outputs normalized coordinates, we need to map to frame space
    
    // Step 1: Rotate detection coordinates (swap x/y)
    float rotated_x = detectionResult.centerY;  // Y becomes X
    float rotated_y = detectionResult.centerX;  // X becomes Y  
    float rotated_w = detectionResult.height;   // Height becomes Width
    float rotated_h = detectionResult.width;    // Width becomes Height
    
    // Step 2: Determine frame size (use max dimension)
    int maxFrameDimension = std::max(width, height);  // max(1280, 720) = 1280
    
    // Step 3: Denormalize to pixel coordinates
    // Note: Detection gives center coords, we need top-left
    float centerX = rotated_x * maxFrameDimension;
    float centerY = rotated_y * maxFrameDimension;
    float boxWidth = rotated_w * maxFrameDimension;
    float boxHeight = rotated_h * maxFrameDimension;
    
    // Convert from center to top-left
    int x_original = static_cast<int>(centerX - boxWidth / 2);
    int y_original = static_cast<int>(centerY - boxHeight / 2);
    int w_original = static_cast<int>(boxWidth);
    int h_original = static_cast<int>(boxHeight);
    
    // Add padding: 100px to height (as per spec) plus 20px more (10px top/bottom)
    h_original += 120;
    
    LOGD("📐 Coordinate conversion per spec:");
    LOGD("📐   Input: norm(%.3f,%.3f,%.3f,%.3f)", 
         detectionResult.centerX, detectionResult.centerY, 
         detectionResult.width, detectionResult.height);
    LOGD("📐   Rotated: (%.3f,%.3f,%.3f,%.3f)",
         rotated_x, rotated_y, rotated_w, rotated_h);
    LOGD("📐   FrameSize: %d (max of %dx%d)", maxFrameDimension, width, height);
    LOGD("📐   Output: (%d,%d,%d,%d) in frame space",
         x_original, y_original, w_original, h_original);
    
    // Clamp to frame bounds
    x_original = std::max(0, x_original);
    y_original = std::max(0, y_original);
    w_original = std::min(w_original, width - x_original);
    h_original = std::min(h_original, height - y_original);
    
    universalscanner::BoundingBox cropBox;
    cropBox.x = x_original;
    cropBox.y = y_original;
    cropBox.width = w_original;
    cropBox.height = h_original;
    
    std::string classType = getClassType(detectionResult.classIndex);
    LOGD("🎯 Final crop region: (%d,%d,%d,%d) for %s",
         cropBox.x, cropBox.y, cropBox.width, cropBox.height, classType.c_str());
    
    // ====== STEP 6: EXTRACT YUV CROP ======
    auto croppedYuv = extractYuvCrop(frameData, frameSize, width, height, cropBox);
    if (croppedYuv.empty()) {
        LOGE("Failed to extract YUV crop");
        return results;
    }
    
    // Debug: Save YUV crop
    LOGD("📸 Debug images enabled: %s", enableDebugImages ? "YES" : "NO");
    if (enableDebugImages) {
        LOGD("📸 Saving OCR debug image: 0_ocr_yuv_crop.jpg");
        // Extract Y plane for visualization
        size_t ySize = cropBox.width * cropBox.height;
        const uint8_t* yPlane = croppedYuv.data();
        const uint8_t* uPlane = croppedYuv.data() + ySize;
        const uint8_t* vPlane = croppedYuv.data() + ySize + ySize/4;
        UniversalScanner::ImageDebugger::saveYUV420("0_ocr_yuv_crop.jpg", 
            yPlane, uPlane, vPlane, cropBox.width, cropBox.height, 
            cropBox.width, (cropBox.width + 1) / 2);
    }
    
    // ====== STEP 7: PROCESS CROP FOR OCR ======
    // Now process this high-res crop through OCR pipeline
    
    if (!ocrEngine_) {
        LOGE("OCR engine not initialized");
        return results;
    }
    
    try {
        // STEP 3: Convert YUV to RGB (at full resolution as per spec)
        int cropW = cropBox.width;
        int cropH = cropBox.height;
        
        LOGD("🔧 Converting YUV to RGB at full resolution %dx%d", cropW, cropH);
        auto rgbVector = yuvConverter->convertYuvToRgb(
            croppedYuv.data(), croppedYuv.size(), cropW, cropH
        );
        
        universalscanner::ImageData rgbImage(cropW, cropH, 3);
        std::copy(rgbVector.begin(), rgbVector.end(), rgbImage.data.begin());
        
        // Debug: Save RGB conversion
        if (enableDebugImages) {
            LOGD("📸 Saving OCR debug image: 1_ocr_rgb_converted.jpg");
            UniversalScanner::ImageDebugger::saveRGB("1_ocr_rgb_converted.jpg", 
                rgbImage.data, cropW, cropH);
        }
        
        // STEP 3 (duplicate numbering in spec): Rotate 90° CW
        LOGD("🔧 Rotating 90° CW for OCR (%dx%d → %dx%d)", cropW, cropH, cropH, cropW);
        universalscanner::ImageData rotatedImage(cropH, cropW, 3);
        for (int y = 0; y < cropH; y++) {
            for (int x = 0; x < cropW; x++) {
                const uint8_t* src = rgbImage.getPixel(x, y);
                // 90° CW: (x,y) → (height-1-y, x)
                uint8_t* dst = rotatedImage.getPixel(cropH - 1 - y, x);
                for (int c = 0; c < 3; c++) {
                    dst[c] = src[c];
                }
            }
        }
        
        // Debug: Save rotated image
        if (enableDebugImages) {
            LOGD("📸 Saving OCR debug image: 2_ocr_rotated.jpg");
            UniversalScanner::ImageDebugger::saveRGB("2_ocr_rotated.jpg", 
                rotatedImage.data, rotatedImage.width, rotatedImage.height);
        }
        
        // STEP 4: Resize to fit 640 on longest dimension
        int rotatedW = rotatedImage.width;
        int rotatedH = rotatedImage.height;
        
        // Find longest dimension and scale to 640
        int targetSize = 640;
        int scaledW, scaledH;
        
        if (rotatedW > rotatedH) {
            // Width is longer, scale to 640 width
            scaledW = targetSize;
            scaledH = (rotatedH * targetSize) / rotatedW;
        } else {
            // Height is longer (or equal), scale to 640 height
            scaledH = targetSize;
            scaledW = (rotatedW * targetSize) / rotatedH;
        }
        
        LOGD("🔧 Resizing %dx%d to %dx%d (longest dimension to 640)", 
             rotatedW, rotatedH, scaledW, scaledH);
        auto scaledImage = rotatedImage.resize(scaledW, scaledH);
        
        // Debug: Save scaled image
        if (enableDebugImages) {
            LOGD("📸 Saving OCR debug image: 3_ocr_scaled.jpg");
            UniversalScanner::ImageDebugger::saveRGB("3_ocr_scaled.jpg", 
                scaledImage.data, scaledW, scaledH);
        }
        
        // STEP 5: Pad to 640x640 with white
        universalscanner::ImageData paddedImage(640, 640, 3);
        std::fill(paddedImage.data.begin(), paddedImage.data.end(), 255);
        
        // Copy scaled image to top-left (as per spec - no centering)
        LOGD("🔧 Padding to 640x640 with white");
        for (int y = 0; y < scaledH; y++) {
            for (int x = 0; x < scaledW; x++) {
                const uint8_t* src = scaledImage.getPixel(x, y);
                uint8_t* dst = paddedImage.getPixel(x, y);
                for (int c = 0; c < 3; c++) {
                    dst[c] = src[c];
                }
            }
        }
        
        // Debug: Save final padded image
        if (enableDebugImages) {
            LOGD("📸 Saving OCR debug image: 4_ocr_final_padded.jpg");
            UniversalScanner::ImageDebugger::saveRGB("4_ocr_final_padded.jpg", 
                paddedImage.data, 640, 640);
        }
        
        LOGD("✅ OCR preprocessing complete: 640x640 padded image");
        
        // Run OCR
        auto ocrResult = ocrEngine_->recognize(paddedImage, classType);
        
        // Create result
        universalscanner::ScanResult result;
        result.type = classType;
        result.value = ocrResult.text;
        result.confidence = ocrResult.confidence;
        result.model = "yolo-ocr-v7-640";
        result.bbox = {
            static_cast<float>(x_original),
            static_cast<float>(y_original),
            static_cast<float>(w_original),
            static_cast<float>(h_original)
        };
        
        results.push_back(result);
        
        LOGD("OCR result: '%s' (conf: %.2f)", result.value.c_str(), result.confidence);
        
    } catch (const std::exception& e) {
        LOGE("OCR processing failed: %s", e.what());
    }
    
    return results;
}


std::string OnnxProcessorV2::getClassType(int classIndex) {
    // Map class index to string type (must match CodeDetectionConstants.h)
    switch (classIndex) {
        case 0: return "code_container_h";      // Horizontal container codes
        case 1: return "code_container_v";      // Vertical container codes  
        case 2: return "code_license_plate";    // Generic license plates
        case 3: return "code_qr_barcode";       // 2D QR codes and barcodes
        case 4: return "code_seal";             // Security seals with serials
        default: return "unknown";
    }
}


std::vector<uint8_t> OnnxProcessorV2::extractYuvCrop(
    const uint8_t* frameData, size_t frameSize,
    int frameWidth, int frameHeight,
    const universalscanner::BoundingBox& bbox
) {
    // Extract a crop from YUV420 data
    size_t ySize = frameWidth * frameHeight;
    size_t uvSize = ySize / 4;
    
    const uint8_t* yPlane = frameData;
    const uint8_t* uPlane = frameData + ySize;
    const uint8_t* vPlane = frameData + ySize + uvSize;
    
    // Calculate crop sizes
    size_t cropYSize = bbox.width * bbox.height;
    size_t cropUVSize = cropYSize / 4;
    size_t totalCropSize = cropYSize + cropUVSize * 2;
    
    std::vector<uint8_t> croppedData(totalCropSize);
    uint8_t* cropY = croppedData.data();
    uint8_t* cropU = croppedData.data() + cropYSize;
    uint8_t* cropV = croppedData.data() + cropYSize + cropUVSize;
    
    // Copy Y plane (full resolution)
    for (int y = 0; y < bbox.height; y++) {
        const uint8_t* srcRow = yPlane + (bbox.y + y) * frameWidth + bbox.x;
        uint8_t* dstRow = cropY + y * bbox.width;
        std::memcpy(dstRow, srcRow, bbox.width);
    }
    
    // Copy U and V planes (half resolution)
    int uvBboxX = bbox.x / 2;
    int uvBboxY = bbox.y / 2;
    int uvBboxW = bbox.width / 2;
    int uvBboxH = bbox.height / 2;
    int uvFrameW = frameWidth / 2;
    
    for (int y = 0; y < uvBboxH; y++) {
        // U plane
        const uint8_t* srcURow = uPlane + (uvBboxY + y) * uvFrameW + uvBboxX;
        uint8_t* dstURow = cropU + y * uvBboxW;
        std::memcpy(dstURow, srcURow, uvBboxW);
        
        // V plane  
        const uint8_t* srcVRow = vPlane + (uvBboxY + y) * uvFrameW + uvBboxX;
        uint8_t* dstVRow = cropV + y * uvBboxW;
        std::memcpy(dstVRow, srcVRow, uvBboxW);
    }
    
    LOGD("✂️ Extracted YUV crop: %dx%d from frame %dx%d", bbox.width, bbox.height, frameWidth, frameHeight);
    return croppedData;
}

} // namespace UniversalScanner