#include "AdaptiveLetterbox.h"
#include <algorithm>

namespace universalscanner {

AdaptiveLetterbox::LetterboxResult AdaptiveLetterbox::letterbox320(
    const ImageData& crop,
    const std::string& classType
) {
    // Always use 320x320 for now - optimize later if needed
    return letterboxToSize(crop, 320, 320);
}

AdaptiveLetterbox::LetterboxResult AdaptiveLetterbox::letterboxToSize(
    const ImageData& input,
    int targetWidth,
    int targetHeight
) {
    AdaptiveLetterbox::LetterboxResult result;
    result.targetSize = targetWidth;  // Assuming square for now
    
    // Calculate scale to fit
    float scaleX = static_cast<float>(targetWidth) / input.width;
    float scaleY = static_cast<float>(targetHeight) / input.height;
    result.scale = std::min(scaleX, scaleY);
    
    // New dimensions
    int newWidth = static_cast<int>(input.width * result.scale);
    int newHeight = static_cast<int>(input.height * result.scale);
    
    // Resize maintaining aspect ratio
    ImageData resized = input.resize(newWidth, newHeight);
    
    // Create letterboxed image (centered padding to match Roboflow)
    result.image = ImageData(targetWidth, targetHeight, input.channels);
    
    // Calculate centered padding
    result.padLeft = (targetWidth - newWidth) / 2;
    result.padTop = (targetHeight - newHeight) / 2;
    
    // Fill with white for OCR (255) instead of black
    std::fill(result.image.data.begin(), result.image.data.end(), 255);
    
    // Copy resized image to center
    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            const uint8_t* srcPixel = resized.getPixel(x, y);
            uint8_t* dstPixel = result.image.getPixel(x + result.padLeft, y + result.padTop);
            
            for (int c = 0; c < input.channels; c++) {
                dstPixel[c] = srcPixel[c];
            }
        }
    }
    
    return result;
}

Point2f AdaptiveLetterbox::mapToOriginal(
    const Point2f& letterboxedPoint,
    const LetterboxResult& letterboxInfo
) {
    // Map from letterboxed coordinates back to original crop coordinates
    Point2f originalPoint;
    originalPoint.x = (letterboxedPoint.x - letterboxInfo.padLeft) / letterboxInfo.scale;
    originalPoint.y = (letterboxedPoint.y - letterboxInfo.padTop) / letterboxInfo.scale;
    return originalPoint;
}

} // namespace universalscanner