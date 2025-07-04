#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace UniversalScanner {

class ImageDebugger {
public:
    // Save raw YUV420 frame data (useful for debugging camera input)
    static bool saveYUV420(const std::string& filename, 
                           const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
                           size_t width, size_t height, 
                           size_t yStride, size_t uvStride);
    
    // Save RGB data as JPEG
    static bool saveRGB(const std::string& filename, 
                       const std::vector<uint8_t>& rgbData, 
                       size_t width, size_t height);
    
    // Save float tensor data as JPEG (converts from CHW format and normalizes)
    static bool saveTensor(const std::string& filename, 
                          const std::vector<float>& tensorData, 
                          size_t width, size_t height);
    
    // Save image with bounding box annotations
    static bool saveAnnotated(const std::string& filename,
                             const std::vector<float>& tensorData,
                             size_t width, size_t height,
                             const std::vector<float>& detections);

private:
    // Ensure debug output directory exists
    static bool ensureDebugDirectory();
    
    // Convert YUV420 to RGB for saving
    static std::vector<uint8_t> yuv420ToRGB(const uint8_t* yPlane, 
                                           const uint8_t* uPlane, 
                                           const uint8_t* vPlane,
                                           size_t width, size_t height,
                                           size_t yStride, size_t uvStride);
    
    // Convert CHW float tensor to HWC uint8 RGB
    static std::vector<uint8_t> tensorToRGB(const std::vector<float>& tensorData, 
                                           size_t width, size_t height);
    
    // Draw bounding box on RGB image
    static void drawBoundingBox(std::vector<uint8_t>& rgbData, 
                               size_t width, size_t height,
                               float centerX, float centerY, 
                               float boxWidth, float boxHeight,
                               uint8_t r = 0, uint8_t g = 255, uint8_t b = 0);
    
    static const std::string DEBUG_DIR;
};

} // namespace UniversalScanner