#include "ImageDebugger.h"

#ifdef ANDROID
#include <android/log.h>
#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "ImageDebugger", fmt, ##__VA_ARGS__)
#else
#import <Foundation/Foundation.h>
#define LOGF(fmt, ...) NSLog(@"ImageDebugger: " fmt, __VA_ARGS__)
#endif

// Enable debug images - force enable for visual debugging
#define DEBUG_IMAGES

// Debug flag detection
#ifdef DEBUG
#define DEBUG_FLAG_DETECTED
#endif

#ifdef DEBUG_IMAGES
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../android/src/main/cpp/lib/stb/stb_image_write.h"
#include <sys/stat.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#endif

namespace UniversalScanner {

const std::string ImageDebugger::DEBUG_DIR = "/sdcard/Download/onnx_debug";

bool ImageDebugger::saveYUV420(const std::string& filename, 
                               const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
                               size_t width, size_t height, 
                               size_t yStride, size_t uvStride) {
#ifdef DEBUG_IMAGES
#ifdef DEBUG_FLAG_DETECTED
    LOGF("DEBUG flag detected in build - debug images enabled");
#else
    LOGF("DEBUG flag NOT detected but debug images force-enabled");
#endif
    if (!ensureDebugDirectory()) {
        return false;
    }
    
    // Convert YUV to RGB first
    auto rgbData = yuv420ToRGB(yPlane, uPlane, vPlane, width, height, yStride, uvStride);
    if (rgbData.empty()) {
        LOGF("Failed to convert YUV to RGB for %s", filename.c_str());
        return false;
    }
    
    // Add timestamp to filename
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H%M%S_");
    
    std::string fullPath = DEBUG_DIR + "/" + ss.str() + filename;
    
    int result = stbi_write_jpg(fullPath.c_str(), width, height, 3, rgbData.data(), 90);
    
    if (result) {
        LOGF("Saved YUV debug image: %s (%zux%zu)", fullPath.c_str(), width, height);
    } else {
        LOGF("Failed to save YUV debug image: %s", fullPath.c_str());
    }
    
    return result != 0;
#else
    LOGF("Debug images disabled - skipping %s", filename.c_str());
    return true;
#endif
}

bool ImageDebugger::saveRGB(const std::string& filename, 
                           const std::vector<uint8_t>& rgbData, 
                           size_t width, size_t height) {
#ifdef DEBUG_IMAGES
    if (!ensureDebugDirectory()) {
        return false;
    }
    
    // Add timestamp to filename
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H%M%S_");
    
    std::string fullPath = DEBUG_DIR + "/" + ss.str() + filename;
    
    int result = stbi_write_jpg(fullPath.c_str(), width, height, 3, rgbData.data(), 90);
    
    if (result) {
        LOGF("Saved RGB debug image: %s (%zux%zu)", fullPath.c_str(), width, height);
    } else {
        LOGF("Failed to save RGB debug image: %s", fullPath.c_str());
    }
    
    return result != 0;
#else
    LOGF("Debug images disabled - skipping %s", filename.c_str());
    return true;
#endif
}

bool ImageDebugger::saveTensor(const std::string& filename, 
                              const std::vector<float>& tensorData, 
                              size_t width, size_t height) {
#ifdef DEBUG_IMAGES
    auto rgbData = tensorToRGB(tensorData, width, height);
    if (rgbData.empty()) {
        LOGF("Failed to convert tensor to RGB for %s", filename.c_str());
        return false;
    }
    
    return saveRGB(filename, rgbData, width, height);
#else
    LOGF("Debug images disabled - skipping %s", filename.c_str());
    return true;
#endif
}

bool ImageDebugger::saveAnnotated(const std::string& filename,
                                 const std::vector<float>& tensorData,
                                 size_t width, size_t height,
                                 const std::vector<float>& detections) {
#ifdef DEBUG_IMAGES
    auto rgbData = tensorToRGB(tensorData, width, height);
    if (rgbData.empty()) {
        LOGF("Failed to convert tensor to RGB for annotation");
        return false;
    }
    
    // Draw bounding boxes on the image
    // detections format: [confidence, x, y, w, h, classIdx] per detection
    for (size_t i = 0; i < detections.size(); i += 6) {
        if (i + 5 < detections.size()) {
            float confidence = detections[i];
            float centerX = detections[i+1] * width;    // Convert normalized to pixels
            float centerY = detections[i+2] * height;   // Convert normalized to pixels
            float boxWidth = detections[i+3] * width;   // Convert normalized to pixels
            float boxHeight = detections[i+4] * height; // Convert normalized to pixels
            
            // Only draw if confidence is above threshold
            if (confidence > 0.25f) {
                drawBoundingBox(rgbData, width, height, centerX, centerY, boxWidth, boxHeight);
                LOGF("Drew bounding box: center=(%.1f,%.1f), size=%.1fx%.1f, conf=%.2f", 
                     centerX, centerY, boxWidth, boxHeight, confidence);
            }
        }
    }
    
    return saveRGB(filename, rgbData, width, height);
#else
    LOGF("Debug images disabled - skipping %s", filename.c_str());
    return true;
#endif
}

bool ImageDebugger::ensureDebugDirectory() {
#ifdef DEBUG_IMAGES
    struct stat st = {0};
    if (stat(DEBUG_DIR.c_str(), &st) == -1) {
        if (mkdir(DEBUG_DIR.c_str(), 0755) != 0) {
            LOGF("Failed to create debug directory: %s", DEBUG_DIR.c_str());
            return false;
        }
        LOGF("Created debug directory: %s", DEBUG_DIR.c_str());
    }
    return true;
#else
    return true;
#endif
}

std::vector<uint8_t> ImageDebugger::yuv420ToRGB(const uint8_t* yPlane, 
                                               const uint8_t* uPlane, 
                                               const uint8_t* vPlane,
                                               size_t width, size_t height,
                                               size_t yStride, size_t uvStride) {
    std::vector<uint8_t> rgbData(width * height * 3);
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Get Y, U, V values
            uint8_t Y = yPlane[y * yStride + x];
            uint8_t U = uPlane[(y/2) * uvStride + (x/2)];
            uint8_t V = vPlane[(y/2) * uvStride + (x/2)];
            
            // YUV to RGB conversion
            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;
            
            int R = (298 * C + 409 * E + 128) >> 8;
            int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
            int B = (298 * C + 516 * D + 128) >> 8;
            
            // Clamp to [0, 255]
            R = R < 0 ? 0 : (R > 255 ? 255 : R);
            G = G < 0 ? 0 : (G > 255 ? 255 : G);
            B = B < 0 ? 0 : (B > 255 ? 255 : B);
            
            // Store in RGB format
            size_t idx = (y * width + x) * 3;
            rgbData[idx] = static_cast<uint8_t>(R);
            rgbData[idx + 1] = static_cast<uint8_t>(G);
            rgbData[idx + 2] = static_cast<uint8_t>(B);
        }
    }
    
    return rgbData;
}

std::vector<uint8_t> ImageDebugger::tensorToRGB(const std::vector<float>& tensorData, 
                                               size_t width, size_t height) {
    if (tensorData.size() != width * height * 3) {
        LOGF("Tensor size mismatch: expected %zu, got %zu", width * height * 3, tensorData.size());
        return {};
    }
    
    std::vector<uint8_t> rgbData(width * height * 3);
    
    // Convert from CHW (Channel-Height-Width) to HWC (Height-Width-Channel)
    // and from float [0,1] to uint8 [0,255]
    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            for (size_t c = 0; c < 3; c++) {
                // CHW: [c][h][w] = c * (height * width) + h * width + w
                size_t chw_idx = c * (height * width) + h * width + w;
                // HWC: [h][w][c] = (h * width + w) * 3 + c
                size_t hwc_idx = (h * width + w) * 3 + c;
                
                float val = tensorData[chw_idx];
                val = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val); // Clamp to [0,1]
                rgbData[hwc_idx] = static_cast<uint8_t>(val * 255.0f);
            }
        }
    }
    
    return rgbData;
}

void ImageDebugger::drawBoundingBox(std::vector<uint8_t>& rgbData, 
                                   size_t width, size_t height,
                                   float centerX, float centerY, 
                                   float boxWidth, float boxHeight,
                                   uint8_t r, uint8_t g, uint8_t b) {
    // Calculate box corners
    int left = static_cast<int>(centerX - boxWidth / 2);
    int right = static_cast<int>(centerX + boxWidth / 2);
    int top = static_cast<int>(centerY - boxHeight / 2);
    int bottom = static_cast<int>(centerY + boxHeight / 2);
    
    // Clamp to image bounds
    left = left < 0 ? 0 : (left >= static_cast<int>(width) ? width - 1 : left);
    right = right < 0 ? 0 : (right >= static_cast<int>(width) ? width - 1 : right);
    top = top < 0 ? 0 : (top >= static_cast<int>(height) ? height - 1 : top);
    bottom = bottom < 0 ? 0 : (bottom >= static_cast<int>(height) ? height - 1 : bottom);
    
    // Draw horizontal lines (top and bottom)
    for (int x = left; x <= right; x++) {
        // Top line
        size_t idx_top = (top * width + x) * 3;
        rgbData[idx_top] = r;
        rgbData[idx_top + 1] = g;
        rgbData[idx_top + 2] = b;
        
        // Bottom line
        size_t idx_bottom = (bottom * width + x) * 3;
        rgbData[idx_bottom] = r;
        rgbData[idx_bottom + 1] = g;
        rgbData[idx_bottom + 2] = b;
    }
    
    // Draw vertical lines (left and right)
    for (int y = top; y <= bottom; y++) {
        // Left line
        size_t idx_left = (y * width + left) * 3;
        rgbData[idx_left] = r;
        rgbData[idx_left + 1] = g;
        rgbData[idx_left + 2] = b;
        
        // Right line
        size_t idx_right = (y * width + right) * 3;
        rgbData[idx_right] = r;
        rgbData[idx_right + 1] = g;
        rgbData[idx_right + 2] = b;
    }
}

} // namespace UniversalScanner