#pragma once

#ifdef __ANDROID__

#include "YuvConverter.h"
#include <jni.h>

namespace UniversalScanner {

/**
 * Android-specific YUV converter using Kotlin/JNI bridge
 * Handles YUV_420_888 format with proper stride and plane handling
 */
class AndroidYuvConverter : public YuvConverter {
private:
    JNIEnv* jniEnv;
    jobject converterInstance;
    jclass converterClass;
    jmethodID convertMethod;
    
public:
    AndroidYuvConverter(JNIEnv* env, jobject context);
    ~AndroidYuvConverter();
    
    std::vector<uint8_t> convertYuvToRgb(
        const uint8_t* frameData,
        size_t frameSize,
        int width,
        int height
    ) override;
    
private:
    bool initializeJavaConverter(jobject context);
};

} // namespace UniversalScanner

#endif // __ANDROID__