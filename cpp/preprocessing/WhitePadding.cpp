#include "WhitePadding.h"
#include <cmath>
#include <cstring>

#ifdef ANDROID
#include <android/log.h>
#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "WhitePadding", fmt, __VA_ARGS__)
#else
#import <Foundation/Foundation.h>
#define LOGF(fmt, ...) NSLog(@"WhitePadding: " fmt, __VA_ARGS__)
#endif

namespace UniversalScanner {

PaddingInfo WhitePadding::calculatePadding(
    size_t inputWidth, size_t inputHeight,
    size_t targetSize
) {
    PaddingInfo info;
    
    // Calculate aspect-ratio preserving scale
    info.scale = std::min(
        static_cast<float>(targetSize) / inputWidth,
        static_cast<float>(targetSize) / inputHeight
    );
    
    // Calculate scaled dimensions
    info.scaledWidth = static_cast<size_t>(inputWidth * info.scale);
    info.scaledHeight = static_cast<size_t>(inputHeight * info.scale);
    
    // Calculate padding (centered)
    info.padLeft = 0;  // Always pad to top-left as per Kotlin implementation
    info.padTop = 0;
    info.padRight = targetSize - info.scaledWidth;
    info.padBottom = targetSize - info.scaledHeight;
    
    LOGF("Padding info: scale=%.3f, scaled=%zux%zu, padding: right=%zu, bottom=%zu",
         info.scale, info.scaledWidth, info.scaledHeight, 
         info.padRight, info.padBottom);
    
    return info;
}

std::vector<float> WhitePadding::applyPadding(
    const std::vector<uint8_t>& rgbData,
    size_t inputWidth, size_t inputHeight,
    size_t targetSize,
    PaddingInfo* info
) {
    const size_t channels = 3;
    
    // Calculate padding dimensions
    PaddingInfo localInfo = calculatePadding(inputWidth, inputHeight, targetSize);
    if (info) {
        *info = localInfo;
    }
    
    // Create white-filled tensor [C, H, W] initialized to 1.0f (white)
    std::vector<float> output(channels * targetSize * targetSize, 1.0f);
    
    // Apply padding and normalization
    padAndNormalize(rgbData.data(), output.data(), 
                    inputWidth, inputHeight, targetSize, localInfo);
    
    return output;
}

void WhitePadding::padAndNormalize(
    const uint8_t* src, float* dst,
    size_t srcWidth, size_t srcHeight,
    size_t dstSize,
    const PaddingInfo& info
) {
    const size_t channels = 3;
    
    // Copy scaled image data to top-left (matching Kotlin drawBitmap at 0,0)
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < info.scaledHeight; ++y) {
            for (size_t x = 0; x < info.scaledWidth; ++x) {
                // Source pixel coordinates with scaling
                float srcY = y / info.scale;
                float srcX = x / info.scale;
                
                // Use bilinear interpolation for smooth scaling
                uint8_t pixelValue = bilinearInterpolate(
                    src, srcX, srcY, srcWidth, srcHeight, c
                );
                
                // Destination: CHW format, top-left aligned
                size_t dstIdx = c * (dstSize * dstSize) + y * dstSize + x;
                
                // Normalize to [0, 1]
                dst[dstIdx] = static_cast<float>(pixelValue) / 255.0f;
            }
        }
    }
}

uint8_t WhitePadding::bilinearInterpolate(
    const uint8_t* src,
    float x, float y,
    size_t width, size_t height,
    size_t channel
) {
    const size_t channels = 3;
    
    // Get integer and fractional parts
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, static_cast<int>(width - 1));
    int y1 = std::min(y0 + 1, static_cast<int>(height - 1));
    
    // Clamp to bounds
    x0 = std::max(0, std::min(x0, static_cast<int>(width - 1)));
    y0 = std::max(0, std::min(y0, static_cast<int>(height - 1)));
    
    float fx = x - x0;
    float fy = y - y0;
    
    // Get pixel values (HWC format in source)
    uint8_t p00 = src[(y0 * width + x0) * channels + channel];
    uint8_t p10 = src[(y0 * width + x1) * channels + channel];
    uint8_t p01 = src[(y1 * width + x0) * channels + channel];
    uint8_t p11 = src[(y1 * width + x1) * channels + channel];
    
    // Bilinear interpolation
    float val = (1 - fx) * (1 - fy) * p00 +
                fx * (1 - fy) * p10 +
                (1 - fx) * fy * p01 +
                fx * fy * p11;
    
    return static_cast<uint8_t>(std::round(val));
}

void WhitePadding::convertCoordinates(
    float paddedX, float paddedY,
    float& originalX, float& originalY,
    const PaddingInfo& info
) {
    // Convert from padded space to original image space
    originalX = (paddedX - info.padLeft) / info.scale;
    originalY = (paddedY - info.padTop) / info.scale;
}

} // namespace UniversalScanner