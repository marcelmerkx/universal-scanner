#pragma once

#ifdef __APPLE__

#include "YuvConverter.h"

namespace UniversalScanner {

/**
 * iOS-specific YUV converter using Core Video frameworks
 * Handles CVPixelBuffer YUV formats with proper color space conversion
 */
class IOSYuvConverter : public YuvConverter {
public:
    IOSYuvConverter();
    ~IOSYuvConverter();
    
    std::vector<uint8_t> convertYuvToRgb(
        const uint8_t* frameData,
        size_t frameSize,
        int width,
        int height
    ) override;
    
    std::vector<uint8_t> convertYuvToGrayscaleRgb(
        const uint8_t* frameData,
        size_t frameSize,
        int width,
        int height
    ) override;
    
private:
    // TODO: Add iOS-specific YUV conversion implementation
    // Will use Core Video frameworks for optimal performance
    void initializeCoreVideoConverter();
};

} // namespace UniversalScanner

#endif // __APPLE__