# Phase 4 & 5 Implementation Plan

## Overview
This document provides a detailed implementation plan for Phases 4 (MLKit Integration) and 5 (Multi-Model Pipeline) of the Universal Scanner, addressing gaps identified in the OCR.md approach.

## Phase 4: MLKit Integration for Text Recognition

### 4.1 Native Bridge Architecture

#### Android (JNI Bridge)
```cpp
// cpp/platform/android/MLKitBridge.h
class MLKitBridge {
public:
    static void initialize(JNIEnv* env, jobject context);
    static std::string recognizeText(const cv::Mat& image);
    static std::string scanBarcode(const cv::Mat& image);
    
private:
    static JavaVM* jvm_;
    static jobject mlKitManager_;
    static jmethodID recognizeTextMethod_;
    static jmethodID scanBarcodeMethod_;
};
```

#### iOS (Obj-C++ Bridge)
```objc
// ios/MLKitBridge.h
@interface MLKitBridge : NSObject
+ (void)initialize;
+ (NSString*)recognizeText:(cv::Mat&)image;
+ (NSString*)scanBarcode:(cv::Mat&)image;
@end
```

### 4.2 MLKit Manager Implementation

#### Android Java Component
```java
// android/src/main/java/com/universal/MLKitManager.java
public class MLKitManager {
    private TextRecognizer textRecognizer;
    private BarcodeScanner barcodeScanner;
    
    public MLKitManager(Context context) {
        // Initialize MLKit components
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        barcodeScanner = BarcodeScanning.getClient();
    }
    
    public String recognizeText(byte[] imageData, int width, int height) {
        // Process with MLKit Text Recognition
    }
    
    public String scanBarcode(byte[] imageData, int width, int height) {
        // Process with MLKit Barcode Scanner
    }
}
```

### 4.3 Integration Points

1. **Initialization**:
   - Load MLKit on app startup (not per-frame)
   - Cache recognizer instances
   - Handle permissions and availability checks

2. **Frame Processing**:
   ```cpp
   // In OnnxProcessor.cpp
   void processDetections(const cv::Mat& frame, const std::vector<Detection>& detections) {
       for (const auto& detection : detections) {
           if (detection.confidence > threshold) {
               cv::Mat crop = extractCrop(frame, detection.bbox);
               
               if (detection.classType == "code_qr_barcode") {
                   std::string result = MLKitBridge::scanBarcode(crop);
               } else if (detection.classType == "text_printed") {
                   std::string result = MLKitBridge::recognizeText(crop);
               }
           }
       }
   }
   ```

3. **Memory Management**:
   - Use cv::Mat for zero-copy image passing
   - Release JNI references properly
   - Implement image buffer pool

### 4.4 Performance Optimizations

1. **Lazy Loading**: Initialize MLKit components only when needed
2. **Batch Processing**: Queue multiple crops for batch recognition
3. **Frame Skipping**: Process every Nth frame for text recognition
4. **Resolution Scaling**: Downscale crops for faster processing

## Phase 5: Multi-Model Detection Pipeline

### 5.1 Pipeline Architecture

```cpp
// cpp/pipeline/MultiModelPipeline.h
class MultiModelPipeline {
public:
    struct PipelineConfig {
        bool enableYOLO = true;
        bool enableMLKitBarcode = true;
        bool enableMLKitText = false;
        float confidenceThreshold = 0.7f;
        int maxResultsPerFrame = 10;
    };
    
    std::vector<ScanResult> process(const cv::Mat& frame, const PipelineConfig& config);
    
private:
    std::unique_ptr<OnnxProcessor> yoloDetector_;
    std::unique_ptr<MLKitProcessor> mlkitProcessor_;
    ResultFusion fusionEngine_;
};
```

### 5.2 Result Fusion Strategy

```cpp
class ResultFusion {
public:
    std::vector<ScanResult> fuseResults(
        const std::vector<YOLODetection>& yoloResults,
        const std::vector<MLKitResult>& mlkitResults
    ) {
        // 1. Deduplicate overlapping detections
        // 2. Prefer higher confidence results
        // 3. Combine complementary detections
        // 4. Apply domain-specific rules
    }
    
private:
    float calculateIoU(const BoundingBox& a, const BoundingBox& b);
    bool shouldMerge(const Detection& a, const Detection& b);
};
```

### 5.3 Parallel Processing Design

```cpp
class ParallelProcessor {
public:
    void processFrame(const cv::Mat& frame) {
        // Dispatch to multiple threads
        auto yoloFuture = std::async(std::launch::async, [&]() {
            return yoloDetector_->detect(frame);
        });
        
        auto mlkitFuture = std::async(std::launch::async, [&]() {
            return mlkitProcessor_->detect(frame);
        });
        
        // Wait and fuse results
        auto yoloResults = yoloFuture.get();
        auto mlkitResults = mlkitFuture.get();
        
        auto fusedResults = fusionEngine_.fuse(yoloResults, mlkitResults);
    }
};
```

### 5.4 Confidence-Based Routing

```cpp
class ConfidenceRouter {
public:
    void routeDetection(const Detection& detection) {
        if (detection.confidence > 0.9) {
            // High confidence - process immediately
            return processImmediate(detection);
        } else if (detection.confidence > 0.7) {
            // Medium confidence - queue for verification
            verificationQueue_.push(detection);
        } else {
            // Low confidence - try alternative models
            alternativeQueue_.push(detection);
        }
    }
};
```

## Implementation Timeline

### Phase 4 (2-3 weeks)
- **Week 1**: 
  - Android JNI bridge implementation
  - MLKitManager Java component
  - Basic text recognition integration
  
- **Week 2**:
  - iOS Obj-C++ bridge
  - Barcode scanning integration
  - Performance optimization
  
- **Week 3**:
  - Testing and debugging
  - Memory leak fixes
  - Documentation

### Phase 5 (2-3 weeks)
- **Week 1**:
  - Multi-model pipeline architecture
  - Result fusion engine
  - Parallel processing framework
  
- **Week 2**:
  - Confidence-based routing
  - Domain-specific processors
  - Performance tuning
  
- **Week 3**:
  - Integration testing
  - Edge case handling
  - Final optimizations

## Key Differences from OCR.md

1. **Integration First**: Phase 4 focuses on deep MLKit integration before building the full OCR pipeline
2. **Parallel Processing**: Phase 5 enables simultaneous YOLO + MLKit detection
3. **Result Fusion**: Intelligent merging of results from multiple models
4. **Performance Focus**: Frame skipping, batch processing, and memory pooling
5. **Native Bridges**: Detailed JNI/Obj-C++ implementation strategy

## Risk Mitigation

1. **MLKit Availability**: Fallback to ONNX OCR if MLKit unavailable
2. **Performance Impact**: Frame rate monitoring and adaptive quality
3. **Memory Pressure**: Implement aggressive buffer recycling
4. **Platform Differences**: Abstract platform-specific code properly

## Success Metrics

- **Phase 4**: MLKit integration working at 15+ FPS
- **Phase 5**: Multi-model fusion with <100ms latency
- **Overall**: 95%+ recognition accuracy on test dataset