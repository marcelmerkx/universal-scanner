#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

namespace UniversalScanner {

struct PaddingInfo {
    float scale;
    size_t scaledWidth;
    size_t scaledHeight;
    size_t padLeft;
    size_t padTop;
    size_t padRight;
    size_t padBottom;
};

class WhitePadding {
public:
    // Apply white padding to maintain aspect ratio
    // Returns normalized float data in CHW format ready for ONNX
    static std::vector<float> applyPadding(
        const std::vector<uint8_t>& rgbData,
        size_t inputWidth, size_t inputHeight,
        size_t targetSize,
        PaddingInfo* info = nullptr
    );
    
    // Calculate padding dimensions without applying
    static PaddingInfo calculatePadding(
        size_t inputWidth, size_t inputHeight,
        size_t targetSize
    );
    
    // Convert coordinates from padded space back to original image space
    static void convertCoordinates(
        float paddedX, float paddedY,
        float& originalX, float& originalY,
        const PaddingInfo& info
    );

private:
    // Apply padding and normalize to [0,1] in CHW format
    static void padAndNormalize(
        const uint8_t* src, float* dst,
        size_t srcWidth, size_t srcHeight,
        size_t dstSize,
        const PaddingInfo& info
    );
    
    // Bilinear interpolation for smooth scaling
    static uint8_t bilinearInterpolate(
        const uint8_t* src,
        float x, float y,
        size_t width, size_t height,
        size_t channel
    );
};

} // namespace UniversalScanner