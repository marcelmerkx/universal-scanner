#pragma once

#include "OnnxProcessor.h"
#include "utils/ScanResult.h"
#include "utils/BoundingBox.h"
#include "utils/ImageData.h"
#include <memory>
#include <map>
#include <functional>

// Forward declarations
namespace universalscanner {
    class YoloOCREngine;
    class ContainerOCRProcessor;
    struct Detection;  // The Detection struct we defined in ContainerOCRProcessor.h
}

namespace UniversalScanner {

/**
 * Enhanced OnnxProcessor with two-stage pipeline support
 * Stage 1: Object detection (existing)
 * Stage 2: OCR recognition (new)
 */
class OnnxProcessorV2 : public OnnxProcessor {
public:
    // Detection result with class type
    struct Detection {
        universalscanner::BoundingBox bbox;      // In original frame coordinates
        std::string classType; // "code_container_v", "code_container_h", etc.
        float confidence;
    };
    
    OnnxProcessorV2();
    ~OnnxProcessorV2() = default;
    
    // Initialize OCR models
    bool initializeOCR(JNIEnv* env, jobject assetManager);
    
    // Process frame with two-stage pipeline
    std::vector<universalscanner::ScanResult> processFrameWithOCR(
        int width, int height, 
        JNIEnv* env, jobject context,
        const uint8_t* frameData, size_t frameSize,
        uint8_t enabledCodeTypesMask = CodeDetectionMask::ALL
    );
    
    // Process OCR using existing detection result (no re-detection)
    std::vector<universalscanner::ScanResult> processOCRWithDetection(
        const DetectionResult& detectionResult,
        int width, int height,
        const uint8_t* frameData, size_t frameSize
    );
    
    // Set OCR model path
    void setOCRModelPath(const std::string& path) { ocrModelPath_ = path; }
    
private:
    // OCR components
    std::unique_ptr<universalscanner::YoloOCREngine> ocrEngine_;
    std::unique_ptr<universalscanner::ContainerOCRProcessor> containerOCRProcessor_;
    std::string ocrModelPath_;
    
    // Forward declare the OCR Detection type for processor functions
    // Processor registry
    using ProcessorFunc = std::function<universalscanner::ScanResult(
        const universalscanner::ImageData&, 
        const universalscanner::Detection&,
        universalscanner::YoloOCREngine&
    )>;
    std::map<std::string, ProcessorFunc> processors_;
    
    // Helper to get class name from index
    std::string getClassType(int classIndex);
    
    // Extract YUV crop from original frame
    std::vector<uint8_t> extractYuvCrop(
        const uint8_t* frameData, 
        size_t frameSize,
        int frameWidth, 
        int frameHeight,
        const universalscanner::BoundingBox& bbox
    );
    
    // Calculate padded bounding box with bounds checking
    universalscanner::BoundingBox getPaddedBox(
        const universalscanner::BoundingBox& bbox,
        int frameWidth = 1280, 
        int frameHeight = 1280,
        int paddingWidth = 20, 
        int paddingHeight = 20
    );
};

} // namespace UniversalScanner