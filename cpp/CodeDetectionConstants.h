#pragma once

#include <cstdint>
#include <string>

namespace UniversalScanner {

/**
 * Code Detection Classes for unified-detection-v7.onnx model
 * 
 * These constants are derived from detection/models/unified-detection-v7.yaml
 * and must match the trained model's class indices exactly.
 * 
 * YAML source:
 * names:
 * - code_container_h     # Index 0
 * - code_container_v     # Index 1  
 * - code_license_plate   # Index 2
 * - code_qr_barcode      # Index 3
 * - code_seal           # Index 4
 * nc: 5
 */

enum class CodeDetectionType : uint8_t {
    CONTAINER_H = 0,      // Horizontal container codes (ISO 6346)
    CONTAINER_V = 1,      // Vertical container codes (ISO 6346)  
    LICENSE_PLATE = 2,    // Generic license plates
    QR_BARCODE = 3,       // 2D QR codes and barcodes
    SEAL = 4              // Security seals with serials
};

// Total number of code detection classes
constexpr size_t CODE_DETECTION_CLASS_COUNT = 5;

// Bitmask definitions for efficient filtering
namespace CodeDetectionMask {
    constexpr uint8_t CONTAINER_H = 0x01;      // bit 0
    constexpr uint8_t CONTAINER_V = 0x02;      // bit 1
    constexpr uint8_t LICENSE_PLATE = 0x04;    // bit 2  
    constexpr uint8_t QR_BARCODE = 0x08;       // bit 3
    constexpr uint8_t SEAL = 0x10;             // bit 4
    constexpr uint8_t ALL = 0x1F;              // All 5 classes enabled
}

/**
 * Convert CodeDetectionType enum to string name
 * Returns the exact string used in YAML model definition
 */
const char* getCodeDetectionClassName(CodeDetectionType type);

/**
 * Convert class index to CodeDetectionType enum
 * Returns CodeDetectionType for valid indices, throws for invalid
 */
CodeDetectionType indexToCodeDetectionType(int classIndex);

/**
 * Convert string name to CodeDetectionType enum
 * Returns CodeDetectionType for valid names, throws for invalid
 */
CodeDetectionType stringToCodeDetectionType(const std::string& className);

/**
 * Convert string name to bitmask value
 * Returns bitmask bit for valid names, 0 for invalid
 */
uint8_t stringToCodeDetectionMask(const std::string& className);

// Compile-time validation that enum size matches model
static_assert(static_cast<size_t>(CodeDetectionType::SEAL) + 1 == CODE_DETECTION_CLASS_COUNT, 
              "CodeDetectionType enum size must match CODE_DETECTION_CLASS_COUNT");

} // namespace UniversalScanner