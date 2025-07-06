#include "CodeDetectionConstants.h"
#include <stdexcept>

namespace UniversalScanner {

const char* getCodeDetectionClassName(CodeDetectionType type) {
    switch (type) {
        case CodeDetectionType::CONTAINER_H:
            return "code_container_h";
        case CodeDetectionType::CONTAINER_V:
            return "code_container_v";
        case CodeDetectionType::LICENSE_PLATE:
            return "code_license_plate";
        case CodeDetectionType::QR_BARCODE:
            return "code_qr_barcode";
        case CodeDetectionType::SEAL:
            return "code_seal";
        default:
            return "unknown";
    }
}

CodeDetectionType indexToCodeDetectionType(int classIndex) {
    if (classIndex < 0 || classIndex >= CODE_DETECTION_CLASS_COUNT) {
        throw std::out_of_range("Invalid code detection class index: " + std::to_string(classIndex));
    }
    return static_cast<CodeDetectionType>(classIndex);
}

CodeDetectionType stringToCodeDetectionType(const std::string& className) {
    if (className == "code_container_h") return CodeDetectionType::CONTAINER_H;
    if (className == "code_container_v") return CodeDetectionType::CONTAINER_V;
    if (className == "code_license_plate") return CodeDetectionType::LICENSE_PLATE;
    if (className == "code_qr_barcode") return CodeDetectionType::QR_BARCODE;
    if (className == "code_seal") return CodeDetectionType::SEAL;
    
    throw std::invalid_argument("Unknown code detection class name: " + className);
}

uint8_t stringToCodeDetectionMask(const std::string& className) {
    if (className == "code_container_h") return CodeDetectionMask::CONTAINER_H;
    if (className == "code_container_v") return CodeDetectionMask::CONTAINER_V;
    if (className == "code_license_plate") return CodeDetectionMask::LICENSE_PLATE;
    if (className == "code_qr_barcode") return CodeDetectionMask::QR_BARCODE;
    if (className == "code_seal") return CodeDetectionMask::SEAL;
    
    // Return 0 for unknown class names (don't throw in bitmask conversion)
    return 0;
}

} // namespace UniversalScanner