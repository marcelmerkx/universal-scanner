#pragma once

#include <onnxruntime_cxx_api.h>
#include "../utils/ImageData.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace universalscanner {

class YoloOCREngine {
public:
    // Character detection from YOLO
    struct CharBox {
        char character;
        float x, y, w, h;     // In letterboxed coordinates (center x, y, width, height)
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
    
    explicit YoloOCREngine(const std::string& modelPath);
    virtual ~YoloOCREngine() = default;
    
    // Main recognition method
    virtual OCRResult recognize(
        const ImageData& letterboxedCrop,  // Already 640x640
        const std::string& classType
    );
    
protected:
    // ONNX model
    std::unique_ptr<Ort::Session> model_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
    
    // Class names
    std::vector<char> classNames_;
    
    // Helper methods
    std::vector<float> preprocessToTensor(const ImageData& image, int modelSize);
    Ort::Value runInference(Ort::Session* session, const std::vector<float>& inputTensor);
    std::vector<CharBox> parseYoloOutput(const Ort::Value& output, int modelSize);
    std::vector<CharBox> runNMS(std::vector<CharBox>& boxes, float iouThreshold);
    std::string assembleText(std::vector<CharBox>& boxes, const std::string& classType);
    float calculateConfidence(const std::vector<CharBox>& boxes);
    
private:
    // Initialize ONNX Runtime
    void initializeOnnx();
    
    // Load model from path
    void loadModel(const std::string& modelPath);
    
    // Calculate IoU for NMS
    float calculateIoU(const CharBox& a, const CharBox& b);
};

} // namespace universalscanner