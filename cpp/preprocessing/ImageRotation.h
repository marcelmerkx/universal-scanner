#pragma once

#include <vector>
#include <cstdint>

namespace UniversalScanner {

class ImageRotation {
public:
    // Rotate RGB image 90° clockwise
    static std::vector<uint8_t> rotate90CW(
        const std::vector<uint8_t>& rgbData,
        size_t width, size_t height
    );
    
    // Rotate RGB image 90° counter-clockwise
    static std::vector<uint8_t> rotate90CCW(
        const std::vector<uint8_t>& rgbData,
        size_t width, size_t height
    );
    
    // Rotate RGB image 180°
    static std::vector<uint8_t> rotate180(
        const std::vector<uint8_t>& rgbData,
        size_t width, size_t height
    );
    
    // Check if rotation is needed based on frame orientation
    // Returns true if frame is 640x480 (portrait), false if 480x640 (landscape)
    static bool needsRotation(size_t width, size_t height) {
        // Only rotate if frame is portrait (height > width)
        // No rotation needed if already landscape (width > height)
        return height > width;
    }
    
    // Get dimensions after 90° rotation
    static void getRotatedDimensions(size_t width, size_t height, size_t& rotatedWidth, size_t& rotatedHeight) {
        rotatedWidth = height;
        rotatedHeight = width;
    }

private:
    // Efficient block-based rotation for cache friendliness
    static void rotateBlock90CW(
        const uint8_t* src, uint8_t* dst,
        size_t srcWidth, size_t srcHeight,
        size_t blockX, size_t blockY,
        size_t blockSize
    );
    
    static void rotateBlock90CCW(
        const uint8_t* src, uint8_t* dst,
        size_t srcWidth, size_t srcHeight,
        size_t blockX, size_t blockY,
        size_t blockSize
    );
};

} // namespace UniversalScanner