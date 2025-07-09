#include "ContainerOCRProcessor.h"
#include <android/log.h>
#include <cctype>
#include <sstream>

#define LOG_TAG "ContainerOCRProcessor"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace universalscanner {

YoloOCREngine::OCRResult ContainerOCRProcessor::recognize(
    const ImageData& letterboxedCrop,
    const std::string& classType
) {
    // Get base OCR result
    auto result = YoloOCREngine::recognize(letterboxedCrop, classType);
    
    // Apply domain-specific corrections
    result.text = applyISO6346Corrections(result.text);
    
    // Validate format
    if (!validateISO6346(result.text)) {
        result.confidence *= 0.7f;  // Reduce confidence for invalid codes
        LOGD("Invalid ISO 6346 format, reduced confidence to %.2f", result.confidence);
    }
    
    return result;
}

ScanResult ContainerOCRProcessor::processContainerCode(
    const ImageData& frame,
    const Detection& detection,
    YoloOCREngine& ocrEngine
) {
    LOGD("🚀 START processContainerCode");
    LOGD("🔍 Processing container code detection: bbox(%d,%d,%d,%d)", 
         detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height);
    
    // Declare variables outside try block
    CropExtractor::CropResult cropResult;
    AdaptiveLetterbox::LetterboxResult letterboxResult;
    YoloOCREngine::OCRResult ocrResult;
    
    try {
        // 1. Extract crop with container-specific padding
        LOGD("🔍 Step 1: Extracting crop...");
        cropResult = CropExtractor::extractCrop(
            frame, detection.bbox, detection.classType
        );
        LOGD("✅ Crop extracted: %dx%d", cropResult.crop.width, cropResult.crop.height);
        
        // 2. Letterbox to 320x320
        LOGD("🔍 Step 2: Applying letterbox...");
        letterboxResult = AdaptiveLetterbox::letterbox320(
            cropResult.crop, detection.classType
        );
        LOGD("✅ Letterbox applied: %dx%d", letterboxResult.image.width, letterboxResult.image.height);
        
        // 3. Run OCR (will apply corrections if using ContainerOCRProcessor)
        LOGD("🔍 Step 3: Running OCR...");
        ocrResult = ocrEngine.recognize(
            letterboxResult.image,
            detection.classType
        );
        LOGD("✅ OCR completed: text='%s'", ocrResult.text.c_str());
        
    } catch (const std::exception& e) {
        LOGE("❌ Exception in processContainerCode: %s", e.what());
        throw;
    }
    
    // 4. Additional validation for container codes
    bool isValid = validateISO6346(ocrResult.text);
    
    // 5. Create scan result
    return createScanResult(
        detection, ocrResult, cropResult, letterboxResult, isValid
    );
}

std::string ContainerOCRProcessor::applyISO6346Corrections(const std::string& raw) {
    if (raw.length() != 11) {
        LOGD("Container code wrong length: %zu (expected 11)", raw.length());
        return raw;
    }
    
    std::string corrected = raw;
    
    // First 4 must be letters (owner code + equipment category)
    for (int i = 0; i < 4; i++) {
        if (std::isdigit(corrected[i])) {
            corrected[i] = digitToLetter(corrected[i]);
        }
    }
    
    // Next 6 must be digits (serial number)
    for (int i = 4; i < 10; i++) {
        if (std::isalpha(corrected[i])) {
            corrected[i] = letterToDigit(corrected[i]);
        }
    }
    
    // Last character is check digit (can be letter or digit)
    // We don't correct it, just validate
    
    LOGD("Container code correction: %s -> %s", raw.c_str(), corrected.c_str());
    return corrected;
}

bool ContainerOCRProcessor::validateISO6346(const std::string& code) {
    if (code.length() != 11) return false;
    
    // Check format: 4 letters + 6 digits + 1 check digit
    for (int i = 0; i < 4; i++) {
        if (!std::isalpha(code[i])) return false;
    }
    
    for (int i = 4; i < 10; i++) {
        if (!std::isdigit(code[i])) return false;
    }
    
    // Calculate and verify check digit
    int calculatedCheck = calculateCheckDigit(code.substr(0, 10));
    
    // Check digit can be 0-9 or 'A' (for check digit 10)
    char expectedCheck = (calculatedCheck == 10) ? 'A' : ('0' + calculatedCheck);
    
    return code[10] == expectedCheck;
}

char ContainerOCRProcessor::digitToLetter(char digit) {
    // Common OCR confusions: 0->O, 1->I, 5->S, 2->Z
    switch(digit) {
        case '0': return 'O';
        case '1': return 'I';
        case '5': return 'S';
        case '2': return 'Z';
        case '8': return 'B';
        case '6': return 'G';
        default: return digit;
    }
}

char ContainerOCRProcessor::letterToDigit(char letter) {
    // Inverse mappings
    switch(letter) {
        case 'O': return '0';
        case 'I': return '1';
        case 'S': return '5';
        case 'Z': return '2';
        case 'B': return '8';
        case 'G': return '6';
        default: return letter;
    }
}

int ContainerOCRProcessor::calculateCheckDigit(const std::string& code) {
    // ISO 6346 check digit calculation
    int sum = 0;
    
    // Letter to number mapping for ISO 6346
    auto letterValue = [](char c) -> int {
        if (std::isdigit(c)) return c - '0';
        // A=10, B=12, C=13... (skip multiples of 11)
        int val = (c - 'A') + 10;
        if (val >= 11) val++;  // Skip 11
        if (val >= 22) val++;  // Skip 22
        if (val >= 33) val++;  // Skip 33
        return val;
    };
    
    // Calculate weighted sum
    for (int i = 0; i < 10; i++) {
        sum += letterValue(code[i]) * (1 << i);  // 2^i
    }
    
    return sum % 11;
}

ScanResult ContainerOCRProcessor::createScanResult(
    const Detection& detection,
    const YoloOCREngine::OCRResult& ocrResult,
    const CropExtractor::CropResult& cropResult,
    const AdaptiveLetterbox::LetterboxResult& letterboxResult,
    bool isValid
) {
    ScanResult result;
    
    // Basic info
    result.type = detection.classType;
    result.value = ocrResult.text;
    result.confidence = ocrResult.confidence;
    result.model = "yolo-ocr-v7-320";
    
    // Bounding box in original frame coordinates
    result.bbox = {
        static_cast<float>(cropResult.originalRect.x),
        static_cast<float>(cropResult.originalRect.y),
        static_cast<float>(cropResult.originalRect.width),
        static_cast<float>(cropResult.originalRect.height)
    };
    
    // Add verbose data
    result.verbose["detectionConfidence"] = detection.confidence;
    result.verbose["ocrConfidence"] = ocrResult.confidence;
    result.verbose["isValidISO6346"] = isValid;
    result.verbose["preprocessMs"] = ocrResult.preprocessMs;
    result.verbose["inferenceMs"] = ocrResult.inferenceMs;
    result.verbose["postprocessMs"] = ocrResult.postprocessMs;
    result.verbose["numCharacters"] = static_cast<int>(ocrResult.characters.size());
    
    // Add character boxes (in crop coordinates)
    std::vector<std::map<std::string, float>> charBoxes;
    for (const auto& charBox : ocrResult.characters) {
        // Map from letterboxed coordinates back to crop coordinates
        auto origPoint = AdaptiveLetterbox::mapToOriginal(
            Point2f(charBox.x, charBox.y),
            letterboxResult
        );
        
        charBoxes.push_back({
            {"char", static_cast<float>(charBox.character)},
            {"x", origPoint.x},
            {"y", origPoint.y},
            {"w", charBox.w / letterboxResult.scale},
            {"h", charBox.h / letterboxResult.scale},
            {"confidence", charBox.confidence}
        });
    }
    result.verbose["characterBoxes"] = charBoxes;
    
    return result;
}

} // namespace universalscanner