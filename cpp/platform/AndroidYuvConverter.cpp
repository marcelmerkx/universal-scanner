#ifdef __ANDROID__

#include "AndroidYuvConverter.h"
#include <android/log.h>

#define LOGF(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "AndroidYuvConverter", fmt, ##__VA_ARGS__)

namespace UniversalScanner {

AndroidYuvConverter::AndroidYuvConverter(JNIEnv* env, jobject context) 
    : jniEnv(env), converterInstance(nullptr), converterClass(nullptr), convertMethod(nullptr) {
    
    if (!initializeJavaConverter(context)) {
        LOGF("Failed to initialize Java YUV converter");
    }
}

AndroidYuvConverter::~AndroidYuvConverter() {
    if (jniEnv && converterInstance) {
        jniEnv->DeleteGlobalRef(converterInstance);
    }
    if (jniEnv && converterClass) {
        jniEnv->DeleteGlobalRef(converterClass);
    }
}

bool AndroidYuvConverter::initializeJavaConverter(jobject context) {
    // Find the YuvConverter class
    jclass localClass = jniEnv->FindClass("com/universal/YuvConverter");
    if (!localClass) {
        LOGF("Could not find YuvConverter class");
        return false;
    }
    
    // Create global reference to the class
    converterClass = (jclass)jniEnv->NewGlobalRef(localClass);
    jniEnv->DeleteLocalRef(localClass);
    
    // Get constructor
    jmethodID constructor = jniEnv->GetMethodID(converterClass, "<init>", "()V");
    if (!constructor) {
        LOGF("Could not find YuvConverter constructor");
        return false;
    }
    
    // Create instance
    jobject localInstance = jniEnv->NewObject(converterClass, constructor);
    if (!localInstance) {
        LOGF("Could not create YuvConverter instance");
        return false;
    }
    
    // Create global reference to the instance
    converterInstance = jniEnv->NewGlobalRef(localInstance);
    jniEnv->DeleteLocalRef(localInstance);
    
    // Get convert method
    convertMethod = jniEnv->GetMethodID(converterClass, "convertYuvToRgb", "([BII)[B");
    if (!convertMethod) {
        LOGF("Could not find convertYuvToRgb method");
        return false;
    }
    
    LOGF("Android YUV converter initialized successfully");
    return true;
}

std::vector<uint8_t> AndroidYuvConverter::convertYuvToRgb(
    const uint8_t* frameData,
    size_t frameSize,
    int width,
    int height
) {
    if (!jniEnv || !converterInstance || !convertMethod) {
        LOGF("YUV converter not properly initialized");
        return {};
    }
    
    try {
        // Create Java byte array from frame data
        jbyteArray javaFrameData = jniEnv->NewByteArray(frameSize);
        if (!javaFrameData) {
            LOGF("Failed to create Java byte array");
            return {};
        }
        
        jniEnv->SetByteArrayRegion(javaFrameData, 0, frameSize, (jbyte*)frameData);
        
        // Call Kotlin conversion method
        jbyteArray rgbResult = (jbyteArray)jniEnv->CallObjectMethod(
            converterInstance, convertMethod, javaFrameData, width, height
        );
        
        // Clean up input array
        jniEnv->DeleteLocalRef(javaFrameData);
        
        if (!rgbResult) {
            LOGF("Kotlin YUV conversion returned null");
            return {};
        }
        
        // Convert Java byte array to C++ vector
        jsize rgbLength = jniEnv->GetArrayLength(rgbResult);
        std::vector<uint8_t> rgbData(rgbLength);
        
        jniEnv->GetByteArrayRegion(rgbResult, 0, rgbLength, (jbyte*)rgbData.data());
        
        // Clean up result array
        jniEnv->DeleteLocalRef(rgbResult);
        
        LOGF("Successfully converted YUV to RGB: %zu bytes -> %zu RGB bytes", frameSize, rgbData.size());
        return rgbData;
        
    } catch (...) {
        LOGF("Exception during YUV to RGB conversion");
        return {};
    }
}

} // namespace UniversalScanner

#endif // __ANDROID__