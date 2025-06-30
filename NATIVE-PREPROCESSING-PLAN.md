# Native C++ Frame Preprocessing Plan

## Overview

Move all frame preprocessing from JavaScript to native C++ to eliminate bridge overhead and coordinate transformation issues. This follows the architecture pattern of react-native-fast-tflite but with proper white padding instead of distortion-causing scaling.

## Current Architecture (Problematic)

```
Camera Frame (YUV420) 
  ↓ JS Bridge
vision-camera-resize-plugin (rotation + scaling)
  ↓ JS Bridge  
C++ ONNX Plugin (HWC→CHW + normalization)
  ↓
ONNX Inference
```

**Issues:**
- Multiple expensive JS bridge crossings
- Coordinate transformation complexity due to intermediate scaling
- Performance lag from bridge overhead
- Width coordinate accuracy issues

## Target Architecture (Optimal)

```
Camera Frame (YUV420)
  ↓ Direct Frame Processor
C++ ONNX Plugin (ALL preprocessing)
  ├── YUV420 → RGB conversion
  ├── 90° clockwise rotation  
  ├── Aspect-ratio preserving white padding
  ├── HWC → CHW transposition
  ├── Normalization to [0,1]
  └── ONNX Inference
```

**Benefits:**
- Zero JS bridge overhead during preprocessing
- Direct coordinate mapping (model space = padded space)
- Consistent with react-native-vision-camera Frame Processor pattern
- Eliminates width correction hacks

## Detailed Implementation Plan

### Phase 1: Frame Access Architecture

#### 1.1 Direct Frame Access Interface
```cpp
// Frame structure from react-native-vision-camera
struct Frame {
  // Platform-specific implementation
  #ifdef ANDROID
    jobject javaFrame;  // android.media.Image
  #else
    CVPixelBufferRef pixelBuffer;  // iOS
  #endif
  
  size_t width;
  size_t height;
  std::string pixelFormat;  // "yuv", "420v", "420f", etc.
  bool isValid;
  
  // Methods to access raw buffer data
  std::vector<uint8_t> getYPlane() const;
  std::vector<uint8_t> getUPlane() const;
  std::vector<uint8_t> getVPlane() const;
};
```

#### 1.2 Frame Processor Plugin Signature Update
```cpp
// Update OnnxPlugin to accept Frame directly
static jsi::Value universalScanner(
    jsi::Runtime& runtime,
    const jsi::Value& thisValue,
    const jsi::Value* arguments,
    size_t count
) {
    // Extract Frame object from arguments[0]
    auto frame = extractFrame(runtime, arguments[0]);
    
    // Extract config from arguments[1] (if provided)
    auto config = count > 1 ? extractConfig(runtime, arguments[1]) : getDefaultConfig();
    
    // Process frame with native preprocessing
    return processFrame(runtime, frame, config);
}
```

### Phase 2: Native YUV to RGB Conversion

#### 2.1 FrameConverter.cpp Implementation
```cpp
#pragma once
#include <vector>
#include <cstdint>

class FrameConverter {
public:
    // Main conversion function
    static std::vector<uint8_t> convertYUVtoRGB(const Frame& frame);
    
private:
    // YUV420 planar format (I420)
    static std::vector<uint8_t> convertI420toRGB(
        const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
        size_t width, size_t height, size_t yStride, size_t uvStride
    );
    
    // YUV420 semi-planar format (NV21/NV12)
    static std::vector<uint8_t> convertNV21toRGB(
        const uint8_t* yPlane, const uint8_t* uvPlane,
        size_t width, size_t height, size_t yStride, size_t uvStride
    );
    
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
};
```

### Phase 3: Native Image Rotation

#### 3.1 ImageRotation.cpp Implementation
```cpp
#pragma once
#include <vector>
#include <cstdint>

class ImageRotation {
public:
    // Rotate RGB image 90° clockwise
    static std::vector<uint8_t> rotate90CW(
        const std::vector<uint8_t>& rgbData,
        size_t width, size_t height
    );
    
    // Check if rotation is needed based on frame orientation
    static bool needsRotation(size_t width, size_t height) {
        // Portrait orientation (height > width) needs rotation for landscape model
        return height > width;
    }
    
private:
    // Efficient block-based rotation for cache friendliness
    static void rotateBlock(
        const uint8_t* src, uint8_t* dst,
        size_t srcWidth, size_t srcHeight,
        size_t blockX, size_t blockY,
        size_t blockSize = 32
    );
};

// Implementation details
std::vector<uint8_t> ImageRotation::rotate90CW(
    const std::vector<uint8_t>& rgbData,
    size_t width, size_t height
) {
    std::vector<uint8_t> rotated(rgbData.size());
    const size_t channels = 3;
    
    // Process in blocks for cache efficiency
    const size_t blockSize = 32;
    for (size_t by = 0; by < height; by += blockSize) {
        for (size_t bx = 0; bx < width; bx += blockSize) {
            rotateBlock(rgbData.data(), rotated.data(), 
                       width, height, bx, by, blockSize);
        }
    }
    
    return rotated;
}
```

### Phase 4: Extract and Modularize White Padding

#### 4.1 WhitePadding.cpp Implementation
```cpp
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

struct PaddingInfo {
    float scale;
    size_t scaledWidth;
    size_t scaledHeight;
    size_t padRight;
    size_t padBottom;
};

class WhitePadding {
public:
    // Apply white padding to maintain aspect ratio
    static std::vector<float> applyPadding(
        const std::vector<uint8_t>& rgbData,
        size_t inputWidth, size_t inputHeight,
        size_t targetSize,
        PaddingInfo* info = nullptr
    );
    
    // Calculate padding dimensions
    static PaddingInfo calculatePadding(
        size_t inputWidth, size_t inputHeight,
        size_t targetSize
    );
    
private:
    // Optimized scaling with bilinear interpolation
    static void scaleAndPad(
        const uint8_t* src, float* dst,
        size_t srcWidth, size_t srcHeight,
        size_t dstWidth, size_t dstHeight,
        float scale
    );
};
```

### Phase 5: Unified Preprocessing Pipeline

#### 5.1 Update OnnxPlugin::processFrame
```cpp
jsi::Value OnnxPlugin::processFrame(
    jsi::Runtime& runtime,
    const Frame& frame,
    const ProcessingConfig& config
) {
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Step 1: YUV to RGB conversion
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> rgbData = FrameConverter::convertYUVtoRGB(frame);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Step 2: Rotation if needed
        if (ImageRotation::needsRotation(frame.width, frame.height)) {
            rgbData = ImageRotation::rotate90CW(rgbData, frame.width, frame.height);
            std::swap(frame.width, frame.height);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        
        // Step 3: White padding + normalization
        PaddingInfo padInfo;
        std::vector<float> tensorData = WhitePadding::applyPadding(
            rgbData, frame.width, frame.height, 640, &padInfo
        );
        auto t4 = std::chrono::high_resolution_clock::now();
        
        // Step 4: Run ONNX inference
        auto* session = static_cast<OnnxSession*>(_session);
        std::vector<float> output = session->runInference(tensorData);
        auto t5 = std::chrono::high_resolution_clock::now();
        
        // Log performance metrics
        if (config.verbose) {
            logf("Native preprocessing times (ms):");
            logf("  YUV→RGB: %.2f", std::chrono::duration<double, std::milli>(t2 - t1).count());
            logf("  Rotation: %.2f", std::chrono::duration<double, std::milli>(t3 - t2).count());
            logf("  Padding: %.2f", std::chrono::duration<double, std::milli>(t4 - t3).count());
            logf("  Inference: %.2f", std::chrono::duration<double, std::milli>(t5 - t4).count());
            logf("  Total: %.2f", std::chrono::duration<double, std::milli>(t5 - start).count());
        }
        
        // Return results with padding info for coordinate mapping
        return createResultObject(runtime, output, padInfo);
        
    } catch (const std::exception& e) {
        throw jsi::JSError(runtime, std::string("Native preprocessing failed: ") + e.what());
    }
}
```

### Phase 6: Platform-Specific Frame Access

#### 6.1 Android Frame Access (JNI)
```cpp
#ifdef ANDROID
Frame extractFrameAndroid(JNIEnv* env, jobject imageObj) {
    Frame frame;
    
    // Get Image class methods
    jclass imageClass = env->GetObjectClass(imageObj);
    jmethodID getWidthMethod = env->GetMethodID(imageClass, "getWidth", "()I");
    jmethodID getHeightMethod = env->GetMethodID(imageClass, "getHeight", "()I");
    jmethodID getFormatMethod = env->GetMethodID(imageClass, "getFormat", "()I");
    jmethodID getPlanesMethod = env->GetMethodID(imageClass, "getPlanes", "()[Landroid/media/Image$Plane;");
    
    frame.width = env->CallIntMethod(imageObj, getWidthMethod);
    frame.height = env->CallIntMethod(imageObj, getHeightMethod);
    
    // Get planes array
    jobjectArray planes = (jobjectArray)env->CallObjectMethod(imageObj, getPlanesMethod);
    
    // Extract Y, U, V planes...
    // [Implementation details]
    
    return frame;
}
#endif
```

#### 6.2 iOS Frame Access (Objective-C++)
```cpp
#ifdef __APPLE__
Frame extractFrameIOS(CVPixelBufferRef pixelBuffer) {
    Frame frame;
    
    frame.width = CVPixelBufferGetWidth(pixelBuffer);
    frame.height = CVPixelBufferGetHeight(pixelBuffer);
    
    // Lock pixel buffer for reading
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    
    // Get Y plane
    void* yPlaneAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    size_t yPlaneSize = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) * frame.height;
    
    // Get UV plane (for NV12 format)
    void* uvPlaneAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
    size_t uvPlaneSize = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1) * (frame.height / 2);
    
    // Copy data...
    // [Implementation details]
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    
    return frame;
}
#endif
```

## Coordinate Transformation Simplification

With native preprocessing, coordinate mapping becomes trivial:

```typescript
// No more correction factors needed!
const scaleX = screenWidth / 640   // Direct 1:1 mapping
const scaleY = screenHeight / 640  // Model space = padded space

const left = (detection.bbox.x - detection.bbox.w / 2) * scaleX
const top = (detection.bbox.y - detection.bbox.h / 2) * scaleY
```

## Camera Resolution Handling

### Dynamic Dimension Detection
```cpp
// Common camera resolutions after 90° rotation
struct CameraResolution {
  size_t width, height;
  const char* name;
};

const CameraResolution KNOWN_RESOLUTIONS[] = {
  {480, 640, "VGA rotated"},
  {720, 960, "HD rotated"}, 
  {1080, 1440, "FHD rotated"},
  {1080, 1920, "FHD rotated wide"},
};
```

### Padding Calculation
```cpp
// Same logic as ImagePaddingModule.kt
float scale = std::min(640.0f / width, 640.0f / height);
size_t scaledWidth = static_cast<size_t>(width * scale);
size_t scaledHeight = static_cast<size_t>(height * scale);
size_t padRight = 640 - scaledWidth;
size_t padBottom = 640 - scaledHeight;
```

## Performance Expectations

### Before (Current)
- Camera frame → JS (12MB @ 60fps = 700MB/s bridge traffic)
- vision-camera-resize-plugin → JS (1.2MB @ 60fps = 70MB/s)
- C++ preprocessing → ONNX
- **Total bridge**: ~770MB/s

### After (Target)  
- Camera frame → C++ directly (zero bridge during preprocessing)
- All processing in native code
- **Total bridge**: ~1KB/s (just detection results)

**Expected improvement**: 99.9% reduction in bridge traffic

## Code Organization

```
cpp/
├── OnnxPlugin.cpp (main entry point, updated for Frame processing)
├── OnnxPlugin.h (updated interface)
├── preprocessing/
│   ├── FrameConverter.cpp (YUV→RGB conversion)
│   ├── FrameConverter.h
│   ├── ImageRotation.cpp (90° rotation logic)
│   ├── ImageRotation.h
│   ├── WhitePadding.cpp (aspect-ratio preserving padding)
│   ├── WhitePadding.h
│   └── FrameExtractor.cpp (platform-specific frame access)
├── platform/
│   ├── android/
│   │   └── AndroidFrameExtractor.cpp (JNI implementation)
│   └── ios/
│       └── IOSFrameExtractor.mm (Objective-C++ implementation)
└── CMakeLists.txt (updated build configuration)
```

## Build Configuration Updates

### CMakeLists.txt
```cmake
# Add new source files
set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/OnnxPlugin.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/OnnxHelpers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/preprocessing/FrameConverter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/preprocessing/ImageRotation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/preprocessing/WhitePadding.cpp
  # ... existing sources
)

# Platform-specific sources
if(ANDROID)
  list(APPEND SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/platform/android/AndroidFrameExtractor.cpp
  )
elseif(APPLE)
  list(APPEND SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/platform/ios/IOSFrameExtractor.mm
  )
endif()
```

## Validation Strategy

1. **Unit Tests**: Test each preprocessing step independently
2. **Visual Validation**: Compare preprocessed frames with current output
3. **Performance Benchmarks**: Measure frame processing time
4. **Coordinate Accuracy**: Verify bounding box alignment with test images

## Container Detection Benefits

With native white padding:
- **Vertical containers**: Full camera height preserved (essential for `code_container_v`)
- **Horizontal containers**: Full camera width preserved 
- **No distortion**: Objects maintain correct aspect ratios
- **Accurate coordinates**: Direct mapping without correction factors

## Implementation Phases

### Phase 1: Foundation (Week 1)
1. **Create preprocessing directory structure**
   - Set up cpp/preprocessing/ and cpp/platform/ directories
   - Update CMakeLists.txt with new source files

2. **Implement FrameConverter.cpp**
   - YUV420 planar (I420) support
   - YUV420 semi-planar (NV21/NV12) support
   - Efficient integer-based color conversion

3. **Test YUV conversion independently**
   - Create unit tests with known YUV data
   - Verify RGB output correctness

### Phase 2: Core Processing (Week 1-2)
4. **Implement ImageRotation.cpp**
   - Block-based rotation for cache efficiency
   - Automatic orientation detection
   - Performance optimization

5. **Extract and refactor WhitePadding.cpp**
   - Move existing padding logic from runInference
   - Create clean interface with PaddingInfo struct
   - Add bilinear interpolation for quality

6. **Platform-specific Frame extractors**
   - Android: JNI implementation for android.media.Image
   - iOS: Objective-C++ for CVPixelBuffer

### Phase 3: Integration (Week 2)
7. **Update OnnxPlugin interface**
   - Modify to accept Frame objects directly
   - Create unified processFrame method
   - Add performance timing and logging

8. **Update JavaScript bindings**
   - Modify frame processor plugin registration
   - Update TypeScript definitions
   - Ensure backward compatibility during transition

### Phase 4: Testing & Optimization (Week 2-3)
9. **Comprehensive testing**
   - Unit tests for each preprocessing component
   - Integration tests with real camera frames
   - Performance benchmarks vs current implementation
   - Visual validation of preprocessing output

10. **Performance optimization**
    - Profile preprocessing pipeline
    - Optimize memory allocations
    - Consider SIMD optimizations for color conversion
    - Implement frame buffer reuse

### Phase 5: Migration & Cleanup (Week 3)
11. **Remove vision-camera-resize-plugin**
    - Update package.json dependencies
    - Remove plugin initialization code
    - Update documentation

12. **Simplify coordinate transformation**
    - Remove width correction factors
    - Update bounding box mapping to use PaddingInfo
    - Test with all supported code types

## Success Criteria

1. **Performance**: < 10ms total preprocessing time at 1080p
2. **Accuracy**: Pixel-perfect coordinate mapping
3. **Memory**: Zero frame copies during preprocessing
4. **Compatibility**: Works on Android 7+ and iOS 13+
5. **Quality**: No visible artifacts from preprocessing

## Risk Mitigation

1. **Platform differences**: Abstract platform code behind clean interfaces
2. **Performance regression**: Keep current implementation until new one is validated
3. **Color space variations**: Support multiple YUV formats with runtime detection
4. **Memory pressure**: Implement buffer pooling for high frame rates

This comprehensive plan transforms the architecture while maintaining stability and performance throughout the migration.