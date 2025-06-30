#include "FrameConverter.h"
#include <cstring>
#include <stdexcept>

#ifdef ANDROID
#include <android/log.h>
#define LOG(msg) __android_log_print(ANDROID_LOG_INFO, "FrameConverter", "%s", msg)
#else
#import <Foundation/Foundation.h>
#define LOG(msg) NSLog(@"FrameConverter: %s", msg)
#endif

namespace UniversalScanner {

std::vector<uint8_t> FrameConverter::convertYUVtoRGB(const Frame& frame) {
    if (!frame.isValid) {
        throw std::runtime_error("Invalid frame");
    }
    
#ifdef ANDROID
    // Android implementation - extract planes from android.media.Image
    JNIEnv* env = frame.env;
    jobject image = frame.javaFrame;
    
    // Get Image class methods
    jclass imageClass = env->GetObjectClass(image);
    jmethodID getPlanesMethod = env->GetMethodID(imageClass, "getPlanes", "()[Landroid/media/Image$Plane;");
    
    // Get planes array
    jobjectArray planes = (jobjectArray)env->CallObjectMethod(image, getPlanesMethod);
    jsize planeCount = env->GetArrayLength(planes);
    
    if (planeCount < 2) {
        throw std::runtime_error("Invalid number of planes: " + std::to_string(planeCount));
    }
    
    // Get Plane class methods
    jclass planeClass = env->FindClass("android/media/Image$Plane");
    jmethodID getBufferMethod = env->GetMethodID(planeClass, "getBuffer", "()Ljava/nio/ByteBuffer;");
    jmethodID getRowStrideMethod = env->GetMethodID(planeClass, "getRowStride", "()I");
    jmethodID getPixelStrideMethod = env->GetMethodID(planeClass, "getPixelStride", "()I");
    
    // Get Y plane
    jobject yPlaneObj = env->GetObjectArrayElement(planes, 0);
    jobject yBuffer = env->CallObjectMethod(yPlaneObj, getBufferMethod);
    jint yRowStride = env->CallIntMethod(yPlaneObj, getRowStrideMethod);
    uint8_t* yData = (uint8_t*)env->GetDirectBufferAddress(yBuffer);
    
    // Determine format based on plane count and pixel stride
    if (planeCount == 3) {
        // I420 format (YUV420 planar)
        jobject uPlaneObj = env->GetObjectArrayElement(planes, 1);
        jobject vPlaneObj = env->GetObjectArrayElement(planes, 2);
        
        jobject uBuffer = env->CallObjectMethod(uPlaneObj, getBufferMethod);
        jobject vBuffer = env->CallObjectMethod(vPlaneObj, getBufferMethod);
        
        jint uvRowStride = env->CallIntMethod(uPlaneObj, getRowStrideMethod);
        
        uint8_t* uData = (uint8_t*)env->GetDirectBufferAddress(uBuffer);
        uint8_t* vData = (uint8_t*)env->GetDirectBufferAddress(vBuffer);
        
        LOG("Converting I420 format");
        return convertI420toRGB(yData, uData, vData, frame.width, frame.height, yRowStride, uvRowStride);
    } else {
        // NV21/NV12 format (YUV420 semi-planar)
        jobject uvPlaneObj = env->GetObjectArrayElement(planes, 1);
        jobject uvBuffer = env->CallObjectMethod(uvPlaneObj, getBufferMethod);
        jint uvRowStride = env->CallIntMethod(uvPlaneObj, getRowStrideMethod);
        // jint uvPixelStride = env->CallIntMethod(uvPlaneObj, getPixelStrideMethod); // Currently unused
        
        uint8_t* uvData = (uint8_t*)env->GetDirectBufferAddress(uvBuffer);
        
        // NV21: V comes before U, NV12: U comes before V
        if (frame.pixelFormat == "nv21") {
            LOG("Converting NV21 format");
            return convertNV21toRGB(yData, uvData, frame.width, frame.height, yRowStride, uvRowStride);
        } else {
            LOG("Converting NV12 format");
            return convertNV12toRGB(yData, uvData, frame.width, frame.height, yRowStride, uvRowStride);
        }
    }
    
#else
    // iOS implementation - extract planes from CVPixelBuffer
    CVPixelBufferRef pixelBuffer = frame.pixelBuffer;
    
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    
    size_t planeCount = CVPixelBufferGetPlaneCount(pixelBuffer);
    
    if (planeCount == 0) {
        // Interleaved format, not supported yet
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        throw std::runtime_error("Interleaved formats not supported");
    }
    
    // Get Y plane
    uint8_t* yData = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    size_t yRowStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    
    std::vector<uint8_t> result;
    
    if (planeCount >= 2) {
        // NV12 format is most common on iOS
        uint8_t* uvData = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
        size_t uvRowStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
        
        LOG("Converting NV12 format (iOS)");
        result = convertNV12toRGB(yData, uvData, frame.width, frame.height, yRowStride, uvRowStride);
    }
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return result;
#endif
}

std::vector<uint8_t> FrameConverter::convertI420toRGB(
    const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
    size_t width, size_t height, size_t yStride, size_t uvStride
) {
    std::vector<uint8_t> rgb(width * height * 3);
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Y is at full resolution
            size_t yIdx = y * yStride + x;
            uint8_t yVal = yPlane[yIdx];
            
            // U and V are at half resolution
            size_t uvY = y / 2;
            size_t uvX = x / 2;
            size_t uvIdx = uvY * uvStride + uvX;
            
            uint8_t uVal = uPlane[uvIdx];
            uint8_t vVal = vPlane[uvIdx];
            
            // Convert to RGB
            size_t rgbIdx = (y * width + x) * 3;
            yuv2rgb(yVal, uVal, vVal, rgb[rgbIdx], rgb[rgbIdx + 1], rgb[rgbIdx + 2]);
        }
    }
    
    return rgb;
}

std::vector<uint8_t> FrameConverter::convertNV21toRGB(
    const uint8_t* yPlane, const uint8_t* uvPlane,
    size_t width, size_t height, size_t yStride, size_t uvStride
) {
    std::vector<uint8_t> rgb(width * height * 3);
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Y is at full resolution
            size_t yIdx = y * yStride + x;
            uint8_t yVal = yPlane[yIdx];
            
            // UV are interleaved at half resolution
            size_t uvY = y / 2;
            size_t uvX = x / 2;
            size_t uvIdx = uvY * uvStride + uvX * 2;
            
            // NV21: V comes first, then U
            uint8_t vVal = uvPlane[uvIdx];
            uint8_t uVal = uvPlane[uvIdx + 1];
            
            // Convert to RGB
            size_t rgbIdx = (y * width + x) * 3;
            yuv2rgb(yVal, uVal, vVal, rgb[rgbIdx], rgb[rgbIdx + 1], rgb[rgbIdx + 2]);
        }
    }
    
    return rgb;
}

std::vector<uint8_t> FrameConverter::convertNV12toRGB(
    const uint8_t* yPlane, const uint8_t* uvPlane,
    size_t width, size_t height, size_t yStride, size_t uvStride
) {
    std::vector<uint8_t> rgb(width * height * 3);
    
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Y is at full resolution
            size_t yIdx = y * yStride + x;
            uint8_t yVal = yPlane[yIdx];
            
            // UV are interleaved at half resolution
            size_t uvY = y / 2;
            size_t uvX = x / 2;
            size_t uvIdx = uvY * uvStride + uvX * 2;
            
            // NV12: U comes first, then V
            uint8_t uVal = uvPlane[uvIdx];
            uint8_t vVal = uvPlane[uvIdx + 1];
            
            // Convert to RGB
            size_t rgbIdx = (y * width + x) * 3;
            yuv2rgb(yVal, uVal, vVal, rgb[rgbIdx], rgb[rgbIdx + 1], rgb[rgbIdx + 2]);
        }
    }
    
    return rgb;
}

} // namespace UniversalScanner