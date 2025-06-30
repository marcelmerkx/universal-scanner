#include "ImageRotation.h"
#include <algorithm>
#include <cstring>

namespace UniversalScanner {

std::vector<uint8_t> ImageRotation::rotate90CW(
    const std::vector<uint8_t>& rgbData,
    size_t width, size_t height
) {
    std::vector<uint8_t> rotated(rgbData.size());
    
    // For 90° clockwise rotation:
    // Source (x, y) -> Destination (height - 1 - y, x)
    // Width and height are swapped in the result
    
    const size_t blockSize = 32; // Process in blocks for cache efficiency
    
    // Process the image in blocks
    for (size_t by = 0; by < height; by += blockSize) {
        for (size_t bx = 0; bx < width; bx += blockSize) {
            rotateBlock90CW(rgbData.data(), rotated.data(), 
                           width, height, bx, by, blockSize);
        }
    }
    
    return rotated;
}

std::vector<uint8_t> ImageRotation::rotate90CCW(
    const std::vector<uint8_t>& rgbData,
    size_t width, size_t height
) {
    std::vector<uint8_t> rotated(rgbData.size());
    
    // For 90° counter-clockwise rotation:
    // Source (x, y) -> Destination (y, width - 1 - x)
    // Width and height are swapped in the result
    
    const size_t blockSize = 32;
    
    for (size_t by = 0; by < height; by += blockSize) {
        for (size_t bx = 0; bx < width; bx += blockSize) {
            rotateBlock90CCW(rgbData.data(), rotated.data(), 
                            width, height, bx, by, blockSize);
        }
    }
    
    return rotated;
}

void ImageRotation::rotateBlock90CW(
    const uint8_t* src, uint8_t* dst,
    size_t srcWidth, size_t srcHeight,
    size_t blockX, size_t blockY,
    size_t blockSize
) {
    const size_t channels = 3;
    size_t dstWidth = srcHeight; // Dimensions are swapped after rotation
    
    size_t actualBlockWidth = std::min(blockSize, srcWidth - blockX);
    size_t actualBlockHeight = std::min(blockSize, srcHeight - blockY);
    
    for (size_t y = 0; y < actualBlockHeight; y++) {
        for (size_t x = 0; x < actualBlockWidth; x++) {
            size_t srcX = blockX + x;
            size_t srcY = blockY + y;
            
            // 90° CW: (x,y) -> (height-1-y, x)
            size_t dstX = srcHeight - 1 - srcY;
            size_t dstY = srcX;
            
            // Calculate indices
            size_t srcIdx = (srcY * srcWidth + srcX) * channels;
            size_t dstIdx = (dstY * dstWidth + dstX) * channels;
            
            // Copy RGB pixels
            dst[dstIdx] = src[srcIdx];         // R
            dst[dstIdx + 1] = src[srcIdx + 1]; // G
            dst[dstIdx + 2] = src[srcIdx + 2]; // B
        }
    }
}

void ImageRotation::rotateBlock90CCW(
    const uint8_t* src, uint8_t* dst,
    size_t srcWidth, size_t srcHeight,
    size_t blockX, size_t blockY,
    size_t blockSize
) {
    const size_t channels = 3;
    size_t dstWidth = srcHeight; // Dimensions are swapped after rotation
    
    size_t actualBlockWidth = std::min(blockSize, srcWidth - blockX);
    size_t actualBlockHeight = std::min(blockSize, srcHeight - blockY);
    
    for (size_t y = 0; y < actualBlockHeight; y++) {
        for (size_t x = 0; x < actualBlockWidth; x++) {
            size_t srcX = blockX + x;
            size_t srcY = blockY + y;
            
            // 90° CCW: (x,y) -> (y, width-1-x)
            size_t dstX = srcY;
            size_t dstY = srcWidth - 1 - srcX;
            
            // Calculate indices
            size_t srcIdx = (srcY * srcWidth + srcX) * channels;
            size_t dstIdx = (dstY * dstWidth + dstX) * channels;
            
            // Copy RGB pixels
            dst[dstIdx] = src[srcIdx];         // R
            dst[dstIdx + 1] = src[srcIdx + 1]; // G
            dst[dstIdx + 2] = src[srcIdx + 2]; // B
        }
    }
}

} // namespace UniversalScanner