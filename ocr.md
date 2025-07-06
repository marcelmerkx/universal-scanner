The Next step!

# OCR Engine Architecture Plan

This document outlines the architecture and design strategy for the OCR phase of our universal scanner. It follows the object detection phase and enables class-specific recognition via modular, pluggable OCR engines.

---

## 🧠 Goals

- Enable class-specific OCR logic (e.g. vertical container codes, printed labels, seals)
- Maintain high frame processing speed by offloading OCR to background threads
- Keep the full-frame image and crop around detection for debugging, archiving, and reprocessing
- Support future expansion with additional OCR methods

---

## 🎯 Flow Summary

1. VisionCamera provides a full frame (1280×720)
2. Frame is resized (e.g. to 320×320) and passed to the YOLOv8n ONNX model
3. On high-confidence detection:
   - Full frame is saved to cache as JPG
   - Crop is extracted from bounding box (with safe padding)
   - An `OCRJob` is dispatched to a background thread
4. OCR engine selection is based on detection class:
   - For container codes → ONNX OCR (CRNN)
   - For generic printed text → ML Kit
   - Future: more engines as needed

---

## 📦 OCR Job Structure

```cpp
struct OCRJob {
  std::string classType;       // e.g. "code_container_v"
  std::string imagePath;       // path to cropped image
  BoundingBox bbox;            // detection box in full frame
  std::string fullFramePath;  // original full frame image path
};
```

---

## 🧩 OCR Engine Interface

```cpp
class OCREngine {
public:
  virtual std::string recognize(const OCRJob& job) = 0;
  virtual ~OCREngine() = default;
};
```

### Example Implementations
```cpp
class MLKitOCREngine : public OCREngine {
public:
  std::string recognize(const OCRJob& job) override;
};

class ContainerOCRViaONNX : public OCREngine {
public:
  std::string recognize(const OCRJob& job) override;
};
```

---

## 🗺️ OCR Engine Registry

```cpp
class OCRRegistry {
public:
  static void registerEngine(const std::string& classType, std::unique_ptr<OCREngine> engine);
  static OCREngine* getEngineFor(const std::string& classType);
};
```

Registered at init time:
```cpp
OCRRegistry::registerEngine("code_container_v", std::make_unique<ContainerOCRViaONNX>());
OCRRegistry::registerEngine("text_printed", std::make_unique<MLKitOCREngine>());
```

---

## 🚀 Dispatching OCR Jobs

From the detector thread:
```cpp
OCRJob job = { classType, cropPath, bbox, framePath };
ocrQueue.push(job); // queue handled in background
```

Background worker:
```cpp
auto* engine = OCRRegistry::getEngineFor(job.classType);
if (engine != nullptr) {
  std::string result = engine->recognize(job);
  // Assemble result DTO with value, confidence, crop path, etc.
}
```

---

## 📋 Output DTO Sample
```json
{
  "type": "code_container_v",
  "value": "MSKU1234567",
  "source": "onnx_crnn_v1",
  "imageCrop": "/cache/crop_abc.jpg",
  "fullFrame": "/cache/frame_abc.jpg",
  "confidence": 0.97
}
```

---

## 💡 Future Directions
- Support multiple OCR engines per type (e.g. fallback or hybrid decoding)
- Benchmark OCR runtimes for class-specific tuning
- Add confidence-based reprocessing if needed

---