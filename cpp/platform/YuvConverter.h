#pragma once

#include <vector>
#include <cstdint>
#include <memory>

#ifdef __ANDROID__
#include <jni.h>
#endif

namespace UniversalScanner {

/**
 * Platform-specific YUV to RGB converter interface
 * Handles the complexity of YUV_420_888 format differences between Android and iOS
 */
class YuvConverter {
public:
    virtual ~YuvConverter() = default;
    
    /**
     * Convert YUV frame data to RGB format
     * @param frameData Raw YUV frame bytes from camera
     * @param frameSize Size of frame data in bytes
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @return RGB data as uint8_t vector (width * height * 3 bytes) or empty if failed
     */
    virtual std::vector<uint8_t> convertYuvToRgb(
        const uint8_t* frameData,
        size_t frameSize,
        int width,
        int height
    ) = 0;
    
    /**
     * Factory method to create platform-specific converter
     */
    static std::unique_ptr<YuvConverter> create(
#ifdef __ANDROID__
        JNIEnv* env = nullptr,
        jobject context = nullptr
#endif
    );
};

} // namespace UniversalScanner