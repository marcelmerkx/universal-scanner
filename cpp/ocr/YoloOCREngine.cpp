#include "YoloOCREngine.h"
#include <algorithm>
#include <numeric>
#include <android/log.h>

#define LOG_TAG "YoloOCREngine"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace universalscanner {

YoloOCREngine::YoloOCREngine(const std::string& modelPath) {
    initializeOnnx();
    loadModel(modelPath);
    
    // Initialize class names (0-9, A-Z)
    for (char c = '0'; c <= '9'; c++) classNames_.push_back(c);
    for (char c = 'A'; c <= 'Z'; c++) classNames_.push_back(c);
}

void YoloOCREngine::initializeOnnx() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloOCR");
    sessionOptions_ = std::make_unique<Ort::SessionOptions>();
    sessionOptions_->SetIntraOpNumThreads(4);
    sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    memoryInfo_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}

void YoloOCREngine::loadModel(const std::string& modelPath) {
    try {
        model_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), *sessionOptions_);
        LOGD("OCR model loaded successfully from: %s", modelPath.c_str());
    } catch (const Ort::Exception& e) {
        LOGE("Failed to load OCR model: %s", e.what());
        throw;
    }
}

YoloOCREngine::OCRResult YoloOCREngine::recognize(
    const ImageData& letterboxedCrop,
    const std::string& classType
) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    LOGD("🔤 OCR recognize: image %dx%d (%zu bytes) for class %s", 
         letterboxedCrop.width, letterboxedCrop.height, letterboxedCrop.data.size(), classType.c_str());
    
    // 1. Convert image to tensor
    LOGD("🔤 Converting image to tensor...");
    auto tensor = preprocessToTensor(letterboxedCrop, 640);
    LOGD("🔤 Tensor created: %zu elements", tensor.size());
    
    auto inferStart = std::chrono::high_resolution_clock::now();
    
    // 2. Run inference
    LOGD("🔤 Running ONNX inference...");
    auto output = runInference(model_.get(), tensor);
    LOGD("🔤 ONNX inference completed");
    
    auto inferEnd = std::chrono::high_resolution_clock::now();
    
    // 3. Parse YOLO output
    LOGD("🔤 Parsing YOLO output...");
    auto boxes = parseYoloOutput(output, 640);
    LOGD("🔤 Found %zu character boxes", boxes.size());
    
    // 4. Apply NMS
    boxes = runNMS(boxes, 0.45f);  // IoU threshold from ContainerCameraApp
    
    // 5. Assemble text
    auto text = assembleText(boxes, classType);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    
    // Calculate timings
    float preprocessMs = std::chrono::duration<float, std::milli>(inferStart - startTime).count();
    float inferenceMs = std::chrono::duration<float, std::milli>(inferEnd - inferStart).count();
    float postprocessMs = std::chrono::duration<float, std::milli>(endTime - inferEnd).count();
    
    return {
        text,
        calculateConfidence(boxes),
        boxes,
        preprocessMs,
        inferenceMs,
        postprocessMs
    };
}

std::vector<float> YoloOCREngine::preprocessToTensor(const ImageData& image, int modelSize) {
    // ImageData is already RGB, just convert to tensor and normalize
    // TODO function to call function, are we sure we want to keep this?
    return image.toTensor();
}

Ort::Value YoloOCREngine::runInference(Ort::Session* session, const std::vector<float>& inputTensor) {
    // Input shape for 640x640 model
    std::array<int64_t, 4> inputShape = {1, 3, 640, 640};
    
    // Create input tensor
    auto inputTensorValue = Ort::Value::CreateTensor<float>(
        *memoryInfo_,
        const_cast<float*>(inputTensor.data()),
        inputTensor.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    // Get input/output names
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};
    
    // Run inference
    auto outputs = session->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensorValue, 1,
        outputNames, 1
    );
    
    return std::move(outputs[0]);
}

std::vector<YoloOCREngine::CharBox> YoloOCREngine::parseYoloOutput(
    const Ort::Value& output,
    int modelSize
) {
    // Output shape: [1, 40, 8400] - batch, attributes, anchors
    const int numAnchors = 8400;
    const int numAttributes = 40;  // 4 bbox + 36 classes
    const int numClasses = 36;     // 0-9, A-Z
    
    // Get tensor info
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
    
    LOGD("Parsed %zu character detections from YOLO output", boxes.size());
    return boxes;
}

std::vector<YoloOCREngine::CharBox> YoloOCREngine::runNMS(
    std::vector<CharBox>& boxes, 
    float iouThreshold
) {
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
    
    LOGD("NMS: %zu boxes -> %zu boxes", boxes.size(), result.size());
    return result;
}

float YoloOCREngine::calculateIoU(const CharBox& a, const CharBox& b) {
    // Convert center format to corner format
    float a_x1 = a.x - a.w / 2;
    float a_y1 = a.y - a.h / 2;
    float a_x2 = a.x + a.w / 2;
    float a_y2 = a.y + a.h / 2;
    
    float b_x1 = b.x - b.w / 2;
    float b_y1 = b.y - b.h / 2;
    float b_x2 = b.x + b.w / 2;
    float b_y2 = b.y + b.h / 2;
    
    // Calculate intersection
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    
    // Calculate union
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;
    
    return (union_area > 0) ? (inter_area / union_area) : 0;
}

std::string YoloOCREngine::assembleText(
    std::vector<CharBox>& boxes,
    const std::string& classType
) {
    // Sort based on code type
    if (classType == "code_container_v") {
        // Vertical container: sort by Y coordinate (top to bottom)
        std::sort(boxes.begin(), boxes.end(),
            [](const CharBox& a, const CharBox& b) { return a.y < b.y; });
        LOGD("Sorting characters vertically (top to bottom) for %s", classType.c_str());
    } else {
        // Horizontal container and others: sort by X coordinate (left to right)
        std::sort(boxes.begin(), boxes.end(),
            [](const CharBox& a, const CharBox& b) { return a.x < b.x; });
        LOGD("Sorting characters horizontally (left to right) for %s", classType.c_str());
    }
    
    // For containers: limit to 11 characters (ISO 6346)
    if (classType.find("container") != std::string::npos && boxes.size() > 11) {
        boxes.resize(11);
    }
    
    // Build text
    std::string text;
    for (const auto& box : boxes) {
        text += box.character;
    }
    
    LOGD("Assembled text: %s", text.c_str());
    return text;
}

float YoloOCREngine::calculateConfidence(const std::vector<CharBox>& boxes) {
    if (boxes.empty()) return 0.0f;
    
    // Average confidence of all detected characters
    float sum = std::accumulate(boxes.begin(), boxes.end(), 0.0f,
        [](float acc, const CharBox& box) { return acc + box.confidence; });
    
    return sum / boxes.size();
}

} // namespace universalscanner