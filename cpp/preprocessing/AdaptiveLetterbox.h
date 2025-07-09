#pragma once

#include <string>
#include "../utils/ImageData.h"

namespace universalscanner {

class AdaptiveLetterbox {
public:
    struct LetterboxResult {
        ImageData image;        // Letterboxed image
        float scale;            // Scale factor applied
        int padLeft;            // Left padding in pixels
        int padTop;             // Top padding in pixels
        int targetSize;         // Model size used (320 or 640)
    };
    
    // Letterbox to 320x320 for all OCR (simplified for initial implementation)
    static LetterboxResult letterbox320(
        const ImageData& crop,
        const std::string& classType
    );
    
    // Direct port from ContainerCameraApp's letterbox logic
    static LetterboxResult letterboxToSize(
        const ImageData& input,
        int targetWidth,
        int targetHeight
    );
    
    // Convert letterboxed coordinates back to original image
    static Point2f mapToOriginal(
        const Point2f& letterboxedPoint,
        const LetterboxResult& letterboxInfo
    );
};

} // namespace universalscanner