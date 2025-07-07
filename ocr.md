# Two-Stage YOLO OCR Architecture - Implementation Blueprint

This document provides a detailed, implementation-ready architecture for the two-stage YOLO OCR pipeline, optimized for 320x320 models and designed to reuse proven ContainerCameraApp components.

---

## 🎯 Core Architecture Decisions

1. **320x320 for BOTH stages**: Detection AND OCR use 320x320 models
   - Crops are typically 150-250px, perfect for 320x320 
   - 43% faster than 640x640 (our benchmarks)
   - Sufficient resolution for character recognition

2. **Adaptive preprocessing**: Smart padding based on crop aspect ratio
3. **Reuse ContainerCameraApp code**: Direct translation of working components
4. **Performance-first design**: Every millisecond counts

---

## 📐 Stage 1: Detection Pipeline (Existing)

```cpp
// cpp/OnnxProcessor.h - Already implemented
class OnnxProcessor {
public:
    struct Detection {
        BoundingBox bbox;      // {x, y, width, height} in original frame coords
        std::string classType; // "code_container_v", "code_container_h", etc.
        float confidence;
    };
    
    std::vector<Detection> detectObjects(
        const cv::Mat& frame,     // 1280x720 original
        int modelSize = 320       // Using 320x320 for speed
    );
};
```

---

## 🔍 Stage 2: OCR Pipeline (New)

### 2.1 Crop Extraction Module

```cpp
// cpp/preprocessing/CropExtractor.h
class CropExtractor {
public:
    struct CropResult {
        cv::Mat crop;           // Extracted crop with padding
        cv::Rect originalRect;  // Original bbox in frame
        float padScale;         // Padding scale applied (typically 1.2-1.5)
    };
    
    // Extract crop with smart padding based on detection type
    static CropResult extractCrop(
        const cv::Mat& frame,           // Original 1280x720 frame
        const BoundingBox& bbox,        // Detection bbox
        const std::string& classType    // For class-specific padding
    ) {
        // Reuse logic from ContainerCameraApp's ImagePaddingModule.kt
        float padScale = getPaddingScale(classType);
        
        // Calculate padded rectangle
        int padX = bbox.width * (padScale - 1.0f) / 2;
        int padY = bbox.height * (padScale - 1.0f) / 2;
        
        cv::Rect paddedRect(
            std::max(0, bbox.x - padX),
            std::max(0, bbox.y - padY),
            std::min(frame.cols - bbox.x + padX, bbox.width + 2 * padX),
            std::min(frame.rows - bbox.y + padY, bbox.height + 2 * padY)
        );
        
        return {frame(paddedRect).clone(), paddedRect, padScale};
    }
    
private:
    static float getPaddingScale(const std::string& classType) {
        if (classType == "code_container_v") return 1.3f;  // More vertical padding
        if (classType == "code_container_h") return 1.2f;  // More horizontal padding
        return 1.25f; // Default
    }
};
```

### 2.2 Adaptive Letterbox Preprocessing

```cpp
// cpp/preprocessing/AdaptiveLetterbox.h
class AdaptiveLetterbox {
public:
    struct LetterboxResult {
        cv::Mat image;          // Letterboxed image
        float scale;            // Scale factor applied
        int padLeft, padTop;    // Padding offsets
        int targetSize;         // Model size used (320 or 640)
    };
    
    // Letterbox to 320x320 for all OCR (simplified for initial implementation)
    static LetterboxResult letterbox320(
        const cv::Mat& crop,
        const std::string& classType
    ) {
        // Always use 320x320 for now - optimize later if needed
        return letterboxToSize(crop, 320, 320);
    }
    
    // Direct port from ContainerCameraApp's letterbox logic
    static LetterboxResult letterboxToSize(
        const cv::Mat& input,
        int targetWidth,
        int targetHeight
    ) {
        // Calculate scale to fit
        float scaleX = targetWidth / (float)input.cols;
        float scaleY = targetHeight / (float)input.rows;
        float scale = std::min(scaleX, scaleY);
        
        // New dimensions
        int newWidth = int(input.cols * scale);
        int newHeight = int(input.rows * scale);
        
        // Resize
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(newWidth, newHeight));
        
        // Create letterboxed image (top-left aligned like ContainerCameraApp)
        cv::Mat letterboxed = cv::Mat::zeros(targetHeight, targetWidth, CV_8UC3);
        resized.copyTo(letterboxed(cv::Rect(0, 0, newWidth, newHeight)));
        
        return {letterboxed, scale, 0, 0, targetWidth};
    }
};
```

### 2.3 YOLO OCR Engine

```cpp
// cpp/ocr/YoloOCREngine.h
class YoloOCREngine {
public:
    // Character detection from YOLO
    struct CharBox {
        char character;
        float x, y, w, h;     // In letterboxed coordinates
        float confidence;
    };
    
    // OCR result with timing
    struct OCRResult {
        std::string text;
        float confidence;
        std::vector<CharBox> characters;
        float preprocessMs;
        float inferenceMs;
        float postprocessMs;
    };
    
    YoloOCREngine(const std::string& modelPath) {
        // Load single 320x320 model
        model_ = loadModel(modelPath);  // e.g., "container-ocr-v7-320.onnx"
        
        // Initialize class names (0-9, A-Z)
        for (char c = '0'; c <= '9'; c++) classNames_.push_back(c);
        for (char c = 'A'; c <= 'Z'; c++) classNames_.push_back(c);
    }
    
    OCRResult recognize(
        const cv::Mat& letterboxedCrop,  // Already 320x320
        const std::string& classType
    ) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 1. Run inference
        auto tensor = preprocessToTensor(letterboxedCrop, 320);
        
        auto inferStart = std::chrono::high_resolution_clock::now();
        auto output = runInference(model_.get(), tensor);  // Returns Ort::Value
        auto inferEnd = std::chrono::high_resolution_clock::now();
        
        // 2. Post-process (reuse ContainerCameraApp logic)
        auto boxes = parseYoloOutput(output, 320);
        boxes = runNMS(boxes, 0.45f);  // IoU threshold from ContainerCameraApp
        
        // 3. Assemble text
        auto text = assembleText(boxes, classType);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        
        return {
            text,
            calculateConfidence(boxes),
            boxes,
            std::chrono::duration<float, std::milli>(inferStart - startTime).count(),
            std::chrono::duration<float, std::milli>(inferEnd - inferStart).count(),
            std::chrono::duration<float, std::milli>(endTime - inferEnd).count()
        };
    }
    
private:
    std::unique_ptr<Ort::Session> model_;
    std::vector<char> classNames_;
    
    // Direct port from ContainerCameraApp's YOLO output parsing
    std::vector<CharBox> parseYoloOutput(
        const Ort::Value& output,
        int modelSize
    ) {
        // Output shape: [1, 40, 8400] - batch, attributes, anchors
        // Attributes: [x, y, w, h, class0...class35]
        const int numAnchors = 8400;
        const int numAttributes = 40;  // 4 bbox + 36 classes
        const int numClasses = 36;     // 0-9, A-Z
        
        // Get raw tensor data
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        const float* data = output.GetTensorData<float>();
        
        // Handle both [40, 8400] and [8400, 40] layouts
        bool attributesFirst = (shape[1] == numAttributes);
        
        std::vector<CharBox> boxes;
        
        for (int i = 0; i < numAnchors; i++) {
            // Get max class probability
            float maxProb = 0;
            int maxClass = 0;
            
            for (int c = 0; c < numClasses; c++) {
                float prob = attributesFirst 
                    ? data[(4 + c) * numAnchors + i]  // [40, 8400] layout
                    : data[i * numAttributes + 4 + c]; // [8400, 40] layout
                    
                if (prob > maxProb) {
                    maxProb = prob;
                    maxClass = c;
                }
            }
            
            if (maxProb > 0.25f) {  // Confidence threshold from ContainerCameraApp
                CharBox box;
                // Extract bbox coordinates
                if (attributesFirst) {
                    box.x = data[0 * numAnchors + i];  // x-center
                    box.y = data[1 * numAnchors + i];  // y-center
                    box.w = data[2 * numAnchors + i];  // width
                    box.h = data[3 * numAnchors + i];  // height
                } else {
                    box.x = data[i * numAttributes + 0];
                    box.y = data[i * numAttributes + 1];
                    box.w = data[i * numAttributes + 2];
                    box.h = data[i * numAttributes + 3];
                }
                box.confidence = maxProb;
                box.character = classNames_[maxClass];
                boxes.push_back(box);
            }
        }
        
        return boxes;
    }
    
    // NMS implementation from ContainerCameraApp
    std::vector<CharBox> runNMS(std::vector<CharBox>& boxes, float iouThreshold) {
        // Sort by confidence
        std::sort(boxes.begin(), boxes.end(), 
            [](const CharBox& a, const CharBox& b) { return a.confidence > b.confidence; });
        
        std::vector<CharBox> result;
        std::vector<bool> suppressed(boxes.size(), false);
        
        for (size_t i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            
            result.push_back(boxes[i]);
            
            for (size_t j = i + 1; j < boxes.size(); j++) {
                if (calculateIoU(boxes[i], boxes[j]) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        
        return result;
    }
    
    // Text assembly with domain corrections
    std::string assembleText(
        std::vector<CharBox>& boxes,
        const std::string& classType
    ) {
        // Sort by X coordinate (left to right)
        std::sort(boxes.begin(), boxes.end(),
            [](const CharBox& a, const CharBox& b) { return a.x < b.x; });
        
        // Take top 11 for containers (ISO 6346)
        if (classType.find("container") != std::string::npos && boxes.size() > 11) {
            boxes.resize(11);
        }
        
        // Build text
        std::string text;
        for (const auto& box : boxes) {
            text += box.character;
        }
        
        return text;
    }
};
```

### 2.4 Domain-Specific OCR Processors

```cpp
// cpp/ocr/ContainerOCRProcessor.h
class ContainerOCRProcessor {
public:
    // Complete processing pipeline for container codes
    static ScanResult processContainerCode(
        const cv::Mat& frame,
        const OnnxProcessor::Detection& detection,
        YoloOCREngine& ocrEngine
    ) {
        // 1. Extract crop with container-specific padding
        auto cropResult = CropExtractor::extractCrop(
            frame, detection.bbox, detection.classType
        );
        
        // 2. Letterbox to 320x320
        auto letterboxResult = AdaptiveLetterbox::letterbox320(
            cropResult.crop, detection.classType
        );
        
        // 3. Run OCR
        auto ocrResult = ocrEngine.recognize(
            letterboxResult.image,  // 320x320 letterboxed image
            detection.classType
        );
        
        // 4. Apply ISO 6346 corrections
        ocrResult.text = applyISO6346Corrections(ocrResult.text);
        
        // 5. Validate
        bool isValid = validateISO6346(ocrResult.text);
        if (!isValid) {
            ocrResult.confidence *= 0.7f;  // Reduce confidence
        }
        
        // 6. Create scan result
        return createScanResult(
            detection, ocrResult, cropResult, letterboxResult, isValid
        );
    }
    
private:
    // Port from ContainerCameraApp's correction logic
    static std::string applyISO6346Corrections(const std::string& raw) {
        if (raw.length() != 11) return raw;
        
        std::string corrected = raw;
        
        // First 4 must be letters (owner code + equipment category)
        for (int i = 0; i < 4; i++) {
            if (std::isdigit(corrected[i])) {
                corrected[i] = digitToLetter(corrected[i]);
            }
        }
        
        // Next 6 must be digits (serial number)
        for (int i = 4; i < 10; i++) {
            if (std::isalpha(corrected[i])) {
                corrected[i] = letterToDigit(corrected[i]);
            }
        }
        
        // Last is check digit (can be letter or digit)
        
        return corrected;
    }
    
    static char digitToLetter(char digit) {
        // Common OCR confusions: 0->O, 1->I, 5->S, 2->Z
        switch(digit) {
            case '0': return 'O';
            case '1': return 'I';
            case '5': return 'S';
            case '2': return 'Z';
            default: return digit;
        }
    }
    
    static char letterToDigit(char letter) {
        // Inverse mappings
        switch(letter) {
            case 'O': return '0';
            case 'I': return '1';
            case 'S': return '5';
            case 'Z': return '2';
            default: return letter;
        }
    }
};
```

### 2.5 Main Processing Pipeline

```cpp
// cpp/UniversalScanner.cpp
class UniversalScanner {
private:
    OnnxProcessor detector_;
    YoloOCREngine ocrEngine_;
    std::map<std::string, std::function<ScanResult(
        const cv::Mat&, 
        const OnnxProcessor::Detection&,
        YoloOCREngine&
    )>> processors_;
    
public:
    UniversalScanner() : 
        detector_("models/unified-detection-v7.onnx"),
        ocrEngine_("models/container-ocr-v7") {
        
        // Register processors
        processors_["code_container_v"] = ContainerOCRProcessor::processContainerCode;
        processors_["code_container_h"] = ContainerOCRProcessor::processContainerCode;
        // Future: processors_["code_qr_barcode"] = QRProcessor::process;
    }
    
    std::vector<ScanResult> processFrame(const cv::Mat& frame) {
        std::vector<ScanResult> results;
        
        // Stage 1: Detection at 320x320
        auto detections = detector_.detectObjects(frame, 320);
        
        // Stage 2: OCR for each detection
        for (const auto& detection : detections) {
            if (detection.confidence < 0.7f) continue;
            
            // Find processor for this class
            auto it = processors_.find(detection.classType);
            if (it != processors_.end()) {
                try {
                    auto result = it->second(frame, detection, ocrEngine_);
                    results.push_back(result);
                } catch (const std::exception& e) {
                    LOG(ERROR) << "OCR failed: " << e.what();
                }
            }
        }
        
        return results;
    }
};
```

### 2.6 Frame Processor Plugin Integration

```cpp
// cpp/UniversalScannerFrameProcessorPlugin.cpp
static jsi::Value universalScanner(
    jsi::Runtime& runtime,
    const jsi::Value& thisValue,
    const jsi::Value* arguments,
    size_t count
) {
    // 1. Extract frame
    auto frameObject = arguments[0].asObject(runtime);
    auto frame = frameConverter.convertToMat(frameObject);
    
    // 2. Extract config
    auto config = arguments[1].asObject(runtime);
    auto enabledTypes = extractEnabledTypes(runtime, config);
    
    // 3. Process synchronously (must complete within frame time)
    static UniversalScanner scanner;  // Singleton for performance
    auto results = scanner.processFrame(frame);
    
    // 4. Filter by enabled types
    results.erase(
        std::remove_if(results.begin(), results.end(),
            [&](const ScanResult& r) {
                return std::find(enabledTypes.begin(), enabledTypes.end(), r.type) == enabledTypes.end();
            }),
        results.end()
    );
    
    // 5. Convert to JSI
    return convertToJSI(runtime, results);
}
```

---

## 📊 Performance Targets & Optimizations

### Model Size Strategy
- **Detection**: Always 320x320 (20 FPS achieved)
- **OCR**: Always 320x320 (can add 640x640 later if accuracy issues)
- **Expected**: All crops use 320x320 initially

### Memory Management
```cpp
class TensorPool {
    std::queue<std::vector<float>> pool320_;
    std::queue<std::vector<float>> pool640_;
    
    std::vector<float> getTensor(int size) {
        auto& pool = (size == 320) ? pool320_ : pool640_;
        if (!pool.empty()) {
            auto tensor = std::move(pool.front());
            pool.pop();
            return tensor;
        }
        return std::vector<float>(3 * size * size);
    }
};
```

### Performance Monitoring
```cpp
struct FrameMetrics {
    float detectionMs;
    float cropExtractionMs;
    float preprocessingMs;
    float ocrInferenceMs;
    float postprocessingMs;
    float totalMs;
    int numDetections;
    int numOCRs;
};
```

---

## 🔧 Components to Reuse from ContainerCameraApp

1. **ImagePaddingModule.kt** → `CropExtractor.cpp`
2. **YoloBridgeModule.kt** → `YoloOCREngine.cpp`
3. **YoloOcrService.ts** → TypeScript API layer
4. **Letterbox logic** → `AdaptiveLetterbox.cpp`
5. **NMS implementation** → Direct port
6. **ISO 6346 corrections** → `ContainerOCRProcessor.cpp`

---

## 🚀 Implementation Steps

1. **Week 1**: Core Infrastructure
   - [ ] Port CropExtractor from ImagePaddingModule
   - [ ] Implement AdaptiveLetterbox with 320/640 selection
   - [ ] Create YoloOCREngine base class
   - [ ] Set up model loading for both sizes

2. **Week 2**: OCR Pipeline
   - [ ] Port YOLO output parsing from ContainerCameraApp
   - [ ] Implement NMS with IoU calculations
   - [ ] Add character assembly logic
   - [ ] Create ContainerOCRProcessor with ISO 6346

3. **Week 3**: Integration & Optimization
   - [ ] Wire up two-stage pipeline in UniversalScanner
   - [ ] Add frame processor plugin binding
   - [ ] Implement memory pooling
   - [ ] Performance profiling and optimization

---

## 🎯 Success Metrics

- **Detection**: 20+ FPS at 320x320 ✓
- **OCR**: 15+ FPS for 320x320 crops
- **End-to-end**: < 100ms per frame with 2-3 detections
- **Accuracy**: 95%+ on container codes
- **Memory**: < 200MB additional for OCR models

---