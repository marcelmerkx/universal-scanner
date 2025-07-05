#pragma once

#include <vector>
#include <memory>

#ifdef __ANDROID__
#include <jni.h>
#endif

/**
 * Abstract interface for platform-specific YUV resizing
 * Enables efficient YUV downsampling before RGB conversion
 */
class IYuvResizer {
public:
    virtual ~IYuvResizer() = default;
    
    /**
     * Resize YUV frame data to target dimensions
     * Maintains proper YUV 4:2:0 subsampling ratios
     * 
     * @param frameData Raw YUV bytes in I420/YUV420p format
     * @param frameSize Size of frame data
     * @param srcWidth Original frame width
     * @param srcHeight Original frame height
     * @param targetWidth Desired output width  
     * @param targetHeight Desired output height
     * @return Resized YUV byte vector or empty on failure
     */
    virtual std::vector<uint8_t> resizeYuv(
        const uint8_t* frameData, 
        size_t frameSize,
        int srcWidth, 
        int srcHeight,
        int targetWidth, 
        int targetHeight
    ) = 0;
};

/**
 * Factory class for creating platform-specific YUV resizers
 */
class YuvResizer {
public:
    /**
     * Create platform-appropriate YUV resizer instance
     */
#ifdef __ANDROID__
    static std::unique_ptr<IYuvResizer> create(JNIEnv* env, jobject context);
#else
    static std::unique_ptr<IYuvResizer> create();
#endif

private:
    YuvResizer() = default;
};