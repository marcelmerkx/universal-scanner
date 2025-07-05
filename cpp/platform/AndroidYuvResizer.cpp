#ifdef __ANDROID__

#include "YuvResizer.h"
#include <android/log.h>
#include <vector>
#include <memory>

#define LOG_TAG "AndroidYuvResizer"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/**
 * Android implementation of YUV resizer using Kotlin/JNI bridge
 * Leverages Android's optimized image processing capabilities
 */
class AndroidYuvResizer : public IYuvResizer {
private:
    JNIEnv* jniEnv;
    jobject resizerInstance;
    jmethodID resizeMethod;

public:
    AndroidYuvResizer(JNIEnv* env, jobject context) : jniEnv(env) {
        // Find the YuvResizer class
        jclass resizerClass = env->FindClass("com/universal/YuvResizer");
        if (!resizerClass) {
            LOGE("Failed to find YuvResizer class");
            resizerInstance = nullptr;
            return;
        }

        // Get constructor
        jmethodID constructor = env->GetMethodID(resizerClass, "<init>", "()V");
        if (!constructor) {
            LOGE("Failed to find YuvResizer constructor");
            resizerInstance = nullptr;
            return;
        }

        // Create YuvResizer instance
        jobject localRef = env->NewObject(resizerClass, constructor);
        if (!localRef) {
            LOGE("Failed to create YuvResizer instance");
            resizerInstance = nullptr;
            return;
        }

        // Create global reference to keep instance alive
        resizerInstance = env->NewGlobalRef(localRef);
        env->DeleteLocalRef(localRef);

        // Get the resizeYuv method
        resizeMethod = env->GetMethodID(resizerClass, "resizeYuv", "([BIIII)[B");
        if (!resizeMethod) {
            LOGE("Failed to find resizeYuv method");
            if (resizerInstance) {
                env->DeleteGlobalRef(resizerInstance);
                resizerInstance = nullptr;
            }
        }

        env->DeleteLocalRef(resizerClass);
        LOGD("AndroidYuvResizer initialized successfully");
    }

    ~AndroidYuvResizer() {
        if (resizerInstance && jniEnv) {
            jniEnv->DeleteGlobalRef(resizerInstance);
        }
    }

    std::vector<uint8_t> resizeYuv(
        const uint8_t* frameData, 
        size_t frameSize,
        int srcWidth, 
        int srcHeight,
        int targetWidth, 
        int targetHeight
    ) override {
        
        if (!resizerInstance || !resizeMethod) {
            LOGE("YuvResizer not properly initialized");
            return {};
        }

        // Create Java byte array from frame data
        jbyteArray javaFrameData = jniEnv->NewByteArray(static_cast<jsize>(frameSize));
        if (!javaFrameData) {
            LOGE("Failed to create Java byte array");
            return {};
        }

        jniEnv->SetByteArrayRegion(javaFrameData, 0, static_cast<jsize>(frameSize), 
                                  reinterpret_cast<const jbyte*>(frameData));

        // Call Kotlin resizeYuv method
        jbyteArray resizedResult = (jbyteArray)jniEnv->CallObjectMethod(
            resizerInstance, 
            resizeMethod, 
            javaFrameData, 
            srcWidth, 
            srcHeight,
            targetWidth,
            targetHeight
        );

        // Clean up input array
        jniEnv->DeleteLocalRef(javaFrameData);

        if (!resizedResult) {
            LOGE("YUV resize failed - Kotlin method returned null");
            return {};
        }

        // Convert result back to C++ vector
        jsize resultLength = jniEnv->GetArrayLength(resizedResult);
        std::vector<uint8_t> resizedData(resultLength);

        jniEnv->GetByteArrayRegion(resizedResult, 0, resultLength, 
                                  reinterpret_cast<jbyte*>(resizedData.data()));

        // Clean up result array  
        jniEnv->DeleteLocalRef(resizedResult);

        LOGD("YUV resize successful: %dx%d -> %dx%d (%d bytes)", 
             srcWidth, srcHeight, targetWidth, targetHeight, resultLength);

        return resizedData;
    }
};

// Factory method implementation
std::unique_ptr<IYuvResizer> YuvResizer::create(JNIEnv* env, jobject context) {
    return std::make_unique<AndroidYuvResizer>(env, context);
}

#endif // __ANDROID__