#include "CropExtractor.h"
#include <algorithm>

namespace universalscanner {

CropExtractor::CropResult CropExtractor::extractCrop(
    const ImageData& frame,
    const BoundingBox& bbox,
    const std::string& classType
) {
    float padScale = getPaddingScale(classType);
    
    // Calculate padded rectangle
    int padX = static_cast<int>(bbox.width * (padScale - 1.0f) / 2);
    int padY = static_cast<int>(bbox.height * (padScale - 1.0f) / 2);
    
    // Add extra 50px horizontal padding for emulator offset compensation
    padX += 50;
    
    // Calculate padded rectangle and ensure bounds
    int x1 = std::max(0, bbox.x - padX);
    int y1 = std::max(0, bbox.y - padY);
    int x2 = std::min(frame.width, bbox.x + bbox.width + padX);
    int y2 = std::min(frame.height, bbox.y + bbox.height + padY);
    
    Rectangle paddedRect(x1, y1, x2 - x1, y2 - y1);
    
    CropExtractor::CropResult result;
    result.crop = frame.crop(paddedRect.x, paddedRect.y, paddedRect.width, paddedRect.height);
    result.originalRect = paddedRect;
    result.padScale = padScale;
    
    return result;
}

float CropExtractor::getPaddingScale(const std::string& classType) {
    // Container-specific padding scales
    if (classType == "code_container_v") {
        return 1.3f;  // More vertical padding for vertical containers
    } else if (classType == "code_container_h") {
        return 1.2f;  // More horizontal padding for horizontal containers
    } else if (classType == "code_qr_barcode") {
        return 1.1f;  // Minimal padding for QR codes
    }
    
    return 1.25f;  // Default padding scale
}

} // namespace universalscanner