#pragma once

#include "YoloOCREngine.h"
#include "../preprocessing/CropExtractor.h"
#include "../preprocessing/AdaptiveLetterbox.h"
#include "../utils/ScanResult.h"
#include "../utils/BoundingBox.h"

// Forward declaration to avoid circular dependency
namespace UniversalScanner {
    class OnnxProcessorV2;
}

namespace universalscanner {

// Simple detection result for OCR processing
struct Detection {
    BoundingBox bbox;
    std::string classType;
    float confidence;
};

class ContainerOCRProcessor : public YoloOCREngine {
public:
    using YoloOCREngine::YoloOCREngine;  // Inherit constructor
    
    // Override to add ISO 6346 corrections
    OCRResult recognize(
        const ImageData& letterboxedCrop,
        const std::string& classType
    ) override;
    
    // Complete processing pipeline for container codes
    static ScanResult processContainerCode(
        const ImageData& frame,
        const Detection& detection,
        YoloOCREngine& ocrEngine
    );
    
private:
    // ISO 6346 specific methods
    static std::string applyISO6346Corrections(const std::string& raw);
    static bool validateISO6346(const std::string& code);
    static char digitToLetter(char digit);
    static char letterToDigit(char letter);
    static int calculateCheckDigit(const std::string& code);
    
    // Create scan result from all processing stages
    static ScanResult createScanResult(
        const Detection& detection,
        const YoloOCREngine::OCRResult& ocrResult,
        const CropExtractor::CropResult& cropResult,
        const AdaptiveLetterbox::LetterboxResult& letterboxResult,
        bool isValid
    );
};

} // namespace universalscanner