#pragma once

#include <string>
#include "../utils/BoundingBox.h"
#include "../utils/ImageData.h"

namespace universalscanner {

class CropExtractor {
public:
    struct CropResult {
        ImageData crop;         // Extracted crop with padding
        Rectangle originalRect; // Original bbox in frame
        float padScale;         // Padding scale applied (typically 1.2-1.5)
    };
    
    // Extract crop with smart padding based on detection type
    static CropResult extractCrop(
        const ImageData& frame,         // Original frame
        const BoundingBox& bbox,        // Detection bbox
        const std::string& classType    // For class-specific padding
    );
    
private:
    static float getPaddingScale(const std::string& classType);
};

} // namespace universalscanner