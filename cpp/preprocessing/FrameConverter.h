#pragma once

#include <vector>
#include <cstdint>
#include <string>

#ifdef ANDROID
#include <jni.h>
#endif

#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#endif

namespace UniversalScanner {

// Frame structure representing camera frame data
struct Frame {
#ifdef ANDROID
    jobject javaFrame;  // android.media.Image
    JNIEnv* env;
#else
    CVPixelBufferRef pixelBuffer;  // iOS
#endif
    
    size_t width;
    size_t height;
    std::string pixelFormat;  // "yuv", "420v", "420f", "nv21", "nv12", etc.
    bool isValid;
};

class FrameConverter {
public:
    // Main conversion function - automatically detects format
    static std::vector<uint8_t> convertYUVtoRGB(const Frame& frame);
    
    // Format-specific conversion functions
    static std::vector<uint8_t> convertI420toRGB(
        const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
        size_t width, size_t height, size_t yStride, size_t uvStride
    );
    
    static std::vector<uint8_t> convertNV21toRGB(
        const uint8_t* yPlane, const uint8_t* uvPlane,
        size_t width, size_t height, size_t yStride, size_t uvStride
    );
    
    static std::vector<uint8_t> convertNV12toRGB(
        const uint8_t* yPlane, const uint8_t* uvPlane,
        size_t width, size_t height, size_t yStride, size_t uvStride
    );

private:
    // Efficient YUV to RGB conversion using integer math
    static inline void yuv2rgb(
        uint8_t y, uint8_t u, uint8_t v,
        uint8_t& r, uint8_t& g, uint8_t& b
    ) {
        int c = y - 16;
        int d = u - 128;
        int e = v - 128;
        
        // ITU-R BT.601 conversion
        int r_val = (298 * c + 409 * e + 128) >> 8;
        int g_val = (298 * c - 100 * d - 208 * e + 128) >> 8;
        int b_val = (298 * c + 516 * d + 128) >> 8;
        
        // Clamp to [0, 255]
        r = static_cast<uint8_t>(std::max(0, std::min(255, r_val)));
        g = static_cast<uint8_t>(std::max(0, std::min(255, g_val)));
        b = static_cast<uint8_t>(std::max(0, std::min(255, b_val)));
    }
    
    // Helper to clamp values
    static inline uint8_t clamp(int val) {
        return static_cast<uint8_t>(std::max(0, std::min(255, val)));
    }
};

} // namespace UniversalScanner