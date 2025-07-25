#include "OnnxProcessorV2.h"
#include "DebugConfig.h"
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

#ifdef NDEBUG
  #define LOGD(...) ((void)0)
#else
  #define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#endif

namespace UniversalScanner {

OnnxProcessorV2::OnnxProcessorV2() : OnnxProcessor() {
    LOGD("OnnxProcessorV2 created with two-stage pipeline support");
}

bool OnnxProcessorV2::initializeOCR(JNIEnv* env, jobject assetManager) {
    LOGD("🔧 Starting OCR initialization...");
    try {
        // Check if OCR model file exists
        std::string ocrModelPath = "/data/data/com.cargosnap.universalscanner/files/container-ocr-bt9wl-v4-24072025.onnx";
        
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
    int width, 
    int height,
    const uint8_t* frameData, size_t frameSize
) {
    std::vector<universalscanner::ScanResult> results;
    
    if (detectionResult.confidence < 0.5f) {
        return results;
    }
    
    // ====== STEP 1: COORDINATE CONVERSION (as per detection_to_ocr_pipeline_fix.md) ======
    // Detection outputs normalized coordinates, we need to map to frame space
    LOGD("📐 Coordinate conversion per spec:");
    LOGD("📐   Input: norm(%.3f,%.3f,%.3f,%.3f)", 
         detectionResult.centerX, detectionResult.centerY, 
         detectionResult.width, detectionResult.height);
    
    // 1: Rotate detection coordinates (swap x/y)
    float rotated_x = detectionResult.centerY;  // Y becomes X
    float rotated_y = detectionResult.centerX;  // X becomes Y  
    float rotated_w = detectionResult.height;   // Height becomes Width
    float rotated_h = detectionResult.width;    // Width becomes Height
    LOGD("📐   Rotated: (%.3f,%.3f,%.3f,%.3f)",
         rotated_x, rotated_y, rotated_w, rotated_h);
    
    // 2: Determine frame size (use max dimension)
    int maxFrameDimension = std::max(width, height);  // max(1280, 720) = 1280
    int virtualPaddingSpace = maxFrameDimension - std::min(width, height);
    LOGD("📐   FrameSize: %d (max of %dx%d) with virtual padding of: %d", maxFrameDimension, width, height, virtualPaddingSpace);
    
    // 3: Denormalize to pixel coordinates
    float centerX = rotated_x * maxFrameDimension;
    float centerY = (1 - rotated_y) * maxFrameDimension - virtualPaddingSpace; // work from the top down, so invert Y
    float boxWidth = rotated_w * maxFrameDimension;
    float boxHeight = rotated_h * maxFrameDimension;
    LOGD("📐   Before padding: (%.0f,%.0f,%.0f,%.0f)", centerX, centerY, boxWidth, boxHeight);

    // ====== STEP 2: CROP FOR OCR (as per detection_to_ocr_pipeline_fix.md) ======
    // Add padding using helper function
    int padding_width = 20; // attention! because of rotation, wider means more height on the frame!
    int padding_height = 20;
 
    int w_padded = boxWidth + padding_width;
    int h_padded = boxHeight + padding_height;

    LOGD("📐   After padding: (%.0f,%.0f,%d,%d) in center frame space",
         centerX, centerY, w_padded, h_padded);

    // box edges for troubleshooting
    float halfW = w_padded / 2.0f;
    float halfH = h_padded / 2.0f;

    int minX = static_cast<int>(centerX - halfW);
    int minY = static_cast<int>(centerY - halfH);
    int maxX = static_cast<int>(centerX + halfW - 1);
    int maxY = static_cast<int>(centerY + halfH - 1);

    LOGD("📐   After padding: (%d,%d,%d,%d) box edges",
         minX, minY, maxX, maxY);

    // Ensure dimensions are even for YUV420 format
    if (w_padded % 2 == 1) w_padded--;
    if (h_padded % 2 == 1) h_padded--;
    int centerX_int = static_cast<int>(centerX);
    int centerY_int = static_cast<int>(centerY);
    if (centerX_int % 2 == 1) centerX_int++;
    if (centerY_int % 2 == 1) centerY_int++;
    
    universalscanner::BoundingBox cropBox;
        cropBox.x = centerX_int - w_padded / 2;
        cropBox.y = centerY_int - h_padded / 2;
        cropBox.width = w_padded;
        cropBox.height = h_padded;
    
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
    if (DebugConfig::getInstance().isDebugImagesEnabled()) {
        std::string typeShort = classType.substr(5); // Remove "code_" prefix
        std::string debugPrefix = typeShort + "_0_ocr_yuv_crop.jpg";
        LOGD("📸 Saving OCR debug image: %s", debugPrefix.c_str());
        // Extract Y plane for visualization
        size_t ySize = cropBox.width * cropBox.height;
        const uint8_t* yPlane = croppedYuv.data();
        const uint8_t* uPlane = croppedYuv.data() + ySize;
        const uint8_t* vPlane = croppedYuv.data() + ySize + ySize/4;
        UniversalScanner::ImageDebugger::saveYUV420(debugPrefix.c_str(), 
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
        if (DebugConfig::getInstance().isDebugImagesEnabled()) {
            std::string typeShort = classType.substr(5); // Remove "code_" prefix
            std::string debugFilename = typeShort + "_1_ocr_rgb_converted.jpg";
            LOGD("📸 Saving OCR debug image: %s", debugFilename.c_str());
            UniversalScanner::ImageDebugger::saveRGB(debugFilename.c_str(), 
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
        if (DebugConfig::getInstance().isDebugImagesEnabled()) {
            std::string typeShort = classType.substr(5); // Remove "code_" prefix
            std::string debugFilename = typeShort + "_2_ocr_rotated.jpg";
            LOGD("📸 Saving OCR debug image: %s", debugFilename.c_str());
            UniversalScanner::ImageDebugger::saveRGB(debugFilename.c_str(), 
                rotatedImage.data, rotatedImage.width, rotatedImage.height);
        }
        
        // STEP 4: Resize to fit 320 on longest dimension
        int rotatedW = rotatedImage.width;
        int rotatedH = rotatedImage.height;
        
        // Find longest dimension and scale to 320
        int targetSize = 320;
        int scaledW, scaledH;
        
        if (rotatedW > rotatedH) {
            // Width is longer, scale to 320 width
            scaledW = targetSize;
            scaledH = (rotatedH * targetSize) / rotatedW;
        } else {
            // Height is longer (or equal), scale to 320 height
            scaledH = targetSize;
            scaledW = (rotatedW * targetSize) / rotatedH;
        }
        
        LOGD("🔧 Resizing %dx%d to %dx%d (longest dimension to 320)", 
             rotatedW, rotatedH, scaledW, scaledH);
        auto scaledImage = rotatedImage.resize(scaledW, scaledH);
        
        // Debug: Save scaled image
        if (DebugConfig::getInstance().isDebugImagesEnabled()) {
            std::string typeShort = classType.substr(5); // Remove "code_" prefix
            std::string debugFilename = typeShort + "_3_ocr_scaled.jpg";
            LOGD("📸 Saving OCR debug image: %s", debugFilename.c_str());
            UniversalScanner::ImageDebugger::saveRGB(debugFilename.c_str(), 
                scaledImage.data, scaledW, scaledH);
        }
        
        // STEP 5: Pad to 320x320 with white (centered)
        universalscanner::ImageData paddedImage(320, 320, 3);
        std::fill(paddedImage.data.begin(), paddedImage.data.end(), 255);
        
        // Calculate padding for centered placement
        int padLeft = (320 - scaledW) / 2;
        int padTop = (320 - scaledH) / 2;
        
        // Copy scaled image to center
        LOGD("🔧 Padding to 320x320 with white (centered at %d,%d)", padLeft, padTop);
        for (int y = 0; y < scaledH; y++) {
            for (int x = 0; x < scaledW; x++) {
                const uint8_t* src = scaledImage.getPixel(x, y);
                uint8_t* dst = paddedImage.getPixel(x + padLeft, y + padTop);
                for (int c = 0; c < 3; c++) {
                    dst[c] = src[c];
                }
            }
        }
        
        // Debug: Save final padded image
        if (DebugConfig::getInstance().isDebugImagesEnabled()) {
            std::string typeShort = classType.substr(5); // Remove "code_" prefix
            std::string debugFilename = typeShort + "_4_ocr_final_padded.jpg";
            LOGD("📸 Saving OCR debug image: %s", debugFilename.c_str());
            UniversalScanner::ImageDebugger::saveRGB(debugFilename.c_str(), 
                paddedImage.data, 320, 320);
        }
        
        LOGD("✅ OCR preprocessing complete: 320x320 padded image");
        
        // Run OCR
        auto ocrResult = ocrEngine_->recognize(paddedImage, classType);
        
        // Create result
        universalscanner::ScanResult result;
        result.type = classType;
        result.value = ocrResult.text;
        result.confidence = ocrResult.confidence;
        result.model = ocrEngine_->getModelFilename();
        result.bbox = {
            static_cast<float>(cropBox.x),
            static_cast<float>(cropBox.y),
            static_cast<float>(cropBox.width),
            static_cast<float>(cropBox.height)
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
    const uint8_t* frameData, 
    size_t frameSize,
    int frameWidth, 
    int frameHeight,
    const universalscanner::BoundingBox& bbox
) {
    // Validate input parameters
    if (!frameData || frameSize == 0 || bbox.width <= 0 || bbox.height <= 0) {
        LOGE("Invalid parameters for YUV crop extraction");
        return std::vector<uint8_t>();
    }
    
    // Ensure crop box is within frame bounds
    if (bbox.x < 0 || bbox.y < 0 || 
        bbox.x + bbox.width > frameWidth || 
        bbox.y + bbox.height > frameHeight) {
        LOGE("Crop box out of bounds: crop(%d,%d,%d,%d) frame(%d,%d)",
             bbox.x, bbox.y, bbox.width, bbox.height, frameWidth, frameHeight);
        return std::vector<uint8_t>();
    }
    
    // Extract a crop from YUV420 data
    size_t ySize = frameWidth * frameHeight;
    size_t uvSize = ySize / 4;
    
    // Validate frame data size
    size_t expectedSize = ySize + uvSize * 2;
    if (frameSize < expectedSize) {
        LOGE("Frame data too small: %zu < %zu", frameSize, expectedSize);
        return std::vector<uint8_t>();
    }
    
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

universalscanner::BoundingBox OnnxProcessorV2::getPaddedBox(
    const universalscanner::BoundingBox& bbox,
    int frameWidth, int frameHeight,
    int paddingWidth, int paddingHeight
) {
    // Note: bbox uses x,y as top-left corner, but we have centerX/centerY
    // So we'll work with center coordinates
    float centerX = bbox.x + bbox.width / 2.0f;
    float centerY = bbox.y + bbox.height / 2.0f;
    
    // Calculate the original edges
    float halfWidth = bbox.width / 2.0f;
    float halfHeight = bbox.height / 2.0f;

    float xMin = centerX - halfWidth - paddingWidth;
    float xMax = centerX + halfWidth + paddingWidth;
    float yMin = centerY - halfHeight - paddingHeight;
    float yMax = centerY + halfHeight + paddingHeight;

    // Clamp to frame bounds
    xMin = std::max(0.0f, xMin);
    yMin = std::max(0.0f, yMin);
    xMax = std::min(static_cast<float>(frameWidth - 1), xMax);
    yMax = std::min(static_cast<float>(frameHeight - 1), yMax);

    // Return new bounding box
    universalscanner::BoundingBox paddedBox;
    paddedBox.x = static_cast<int>(xMin);
    paddedBox.y = static_cast<int>(yMin);
    paddedBox.width = static_cast<int>(xMax - xMin);
    paddedBox.height = static_cast<int>(yMax - yMin);
    
    return paddedBox;
}

} // namespace UniversalScanner