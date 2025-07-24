#include "YoloOCREngine.h"
#include <algorithm>
#include <numeric>
#include <android/log.h>

#define LOG_TAG "YoloOCREngine"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace universalscanner {

YoloOCREngine::YoloOCREngine(const std::string& modelPath) {
    // Extract filename from path
    size_t lastSlash = modelPath.find_last_of("/\\");
    modelFilename_ = (lastSlash != std::string::npos) ? modelPath.substr(lastSlash + 1) : modelPath;
    
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
    auto tensor = preprocessToTensor(letterboxedCrop, 320);
    LOGD("🔤 Tensor created: %zu elements", tensor.size());
    
    auto inferStart = std::chrono::high_resolution_clock::now();
    
    // 2. Run inference
    LOGD("🔤 Running ONNX inference...");
    auto output = runInference(model_.get(), tensor);
    LOGD("🔤 ONNX inference completed");
    
    auto inferEnd = std::chrono::high_resolution_clock::now();
    
    // 3. Parse YOLO output
    LOGD("🔤 Parsing YOLO output...");
    auto boxes = parseYoloOutput(output, 320);
    LOGD("🔤 Found %zu character boxes", boxes.size());
    
    // 4. Apply NMS
    boxes = runNMS(boxes, 0.3f);  // Lower IoU threshold for text (characters can be close)
    
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
    // Input shape for 320x320 model
    std::array<int64_t, 4> inputShape = {1, 3, 320, 320};
    
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
    // Expected values
    const int numAttributes = 40;  // 4 bbox + 36 classes
    const int numClasses = 36;     // 0-9, A-Z
    
    // Get tensor info
    auto tensorInfo = output.GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();
    const float* data = output.GetTensorData<float>();
    
    // Validate shape
    if (shape.size() != 3) {
        LOGE("Invalid output tensor dimensions: %zu (expected 3)", shape.size());
        return {};
    }
    
    LOGD("YOLO output shape: [%lld, %lld, %lld]", shape[0], shape[1], shape[2]);
    
    // Dynamically determine anchors and layout
    int numAnchors;
    bool attributesFirst;
    
    if (shape[1] == numAttributes) {
        // [1, 40, N] layout
        attributesFirst = true;
        numAnchors = shape[2];
    } else if (shape[2] == numAttributes) {
        // [1, N, 40] layout
        attributesFirst = false;
        numAnchors = shape[1];
    } else {
        LOGE("Invalid output shape: neither dimension matches expected attributes %d", numAttributes);
        return {};
    }
    
    LOGD("Detected %d anchors, attributes first: %s", numAnchors, attributesFirst ? "true" : "false");
    
    // Validate data size
    size_t expectedSize = shape[0] * shape[1] * shape[2];
    size_t actualSize = tensorInfo.GetElementCount();
    if (actualSize != expectedSize) {
        LOGE("Tensor size mismatch: expected %zu, got %zu", expectedSize, actualSize);
        return {};
    }
    
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
        
        if (maxProb > 0.4f) {  // Confidence threshold (lowered for better detection)
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
            
            LOGD("Character detected: '%c' conf=%.3f at (%.1f,%.1f) size=%.1fx%.1f", 
                 box.character, box.confidence, box.x, box.y, box.w, box.h);
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
    
    // Log surviving characters for debugging
    if (result.size() > 0) {
        std::string chars;
        std::vector<float> xCoords;
        for (const auto& box : result) {
            chars += box.character;
            chars += " ";
            xCoords.push_back(box.x);
        }
        LOGD("Surviving characters after NMS: %s", chars.c_str());
        
        // Log X coordinates for debugging
        std::string xStr;
        for (float x : xCoords) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f ", x);
            xStr += buf;
        }
        LOGD("Character X-coordinates: %s", xStr.c_str());
    }
    
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

std::string YoloOCREngine::assembleHorizontalContainerText(std::vector<CharBox>& boxes) {
    if (boxes.empty()) return "";
    
    // Group characters by lines (Y coordinate)
    // First, check if all characters are likely on a single line
    float minY = boxes[0].y, maxY = boxes[0].y;
    float avgHeight = 0.0f;
    for (const auto& box : boxes) {
        minY = std::min(minY, box.y);
        maxY = std::max(maxY, box.y);
        avgHeight += box.h;
    }
    avgHeight /= boxes.size();
    
    float yRange = maxY - minY;
    LOGD("Y-coordinate range: %.1f (min=%.1f, max=%.1f), avg char height=%.1f", 
         yRange, minY, maxY, avgHeight);
    
    // If Y-range is less than average character height, treat as single line
    // Otherwise use a more generous tolerance for line grouping
    float lineToleranceRatio = (yRange < avgHeight) ? 1.5f : 0.8f; // More generous tolerance
    float lineTolerance = avgHeight * lineToleranceRatio;
    
    // Group characters into lines
    std::vector<std::vector<CharBox>> lines;
    for (const auto& box : boxes) {
        bool addedToLine = false;
        
        // Try to add to existing line
        for (auto& line : lines) {
            if (!line.empty()) {
                float lineY = line[0].y; // Use first character's Y as line reference
                if (std::abs(box.y - lineY) <= lineTolerance) {
                    line.push_back(box);
                    addedToLine = true;
                    break;
                }
            }
        }
        
        // Create new line if not added to existing
        if (!addedToLine) {
            lines.push_back({box});
        }
    }
    
    LOGD("Grouped %zu characters into %zu lines for horizontal container", boxes.size(), lines.size());
    
    // Debug: Log each line's content
    for (size_t i = 0; i < lines.size(); i++) {
        std::string lineChars;
        for (const auto& box : lines[i]) {
            lineChars += box.character;
        }
        LOGD("Line %zu: '%s' (%zu chars at y≈%.1f)", i, lineChars.c_str(), 
             lines[i].size(), lines[i][0].y);
    }
    
    // Check if we should merge all lines into one
    // This happens when the Y-range is small relative to character height
    if (lines.size() > 1 && yRange < avgHeight * 1.5f) {
        LOGD("Y-range (%.1f) suggests single line, merging %zu lines", yRange, lines.size());
        std::vector<CharBox> mergedLine;
        for (const auto& line : lines) {
            mergedLine.insert(mergedLine.end(), line.begin(), line.end());
        }
        lines.clear();
        lines.push_back(mergedLine);
    }
    
    // Sort lines by Y coordinate (top to bottom)
    std::sort(lines.begin(), lines.end(),
        [](const std::vector<CharBox>& a, const std::vector<CharBox>& b) {
            return a[0].y < b[0].y;
        });
    
    // Sort characters within each line by X coordinate (left to right)
    for (auto& line : lines) {
        std::sort(line.begin(), line.end(),
            [](const CharBox& a, const CharBox& b) { return a.x < b.x; });
    }
    
    // Strategy: Be greedy on the top line, only use lower lines if needed
    std::string result;
    
    if (!lines.empty()) {
        // Always use the first (top) line
        std::string topLine;
        for (const auto& box : lines[0]) {
            topLine += box.character;
        }
        result = topLine;
        
        LOGD("Top line: '%s' (%zu chars)", topLine.c_str(), topLine.length());
        
        // Only use additional lines if top line is insufficient for container code
        // Container codes need at least 4 characters (owner code like "ABCU")
        // and ideally 11 characters total (full ISO 6346)
        if (topLine.length() < 4 && lines.size() > 1) {
            LOGD("Top line insufficient (%zu chars), checking second line", topLine.length());
            
            std::string secondLine;
            for (const auto& box : lines[1]) {
                secondLine += box.character;
            }
            
            LOGD("Second line: '%s' (%zu chars)", secondLine.c_str(), secondLine.length());
            
            // Combine lines, but limit total to 11 characters
            std::string combined = topLine + secondLine;
            if (combined.length() > 11) {
                combined = combined.substr(0, 11);
            }
            result = combined;
            
            LOGD("Combined result: '%s'", result.c_str());
        }
    }
    
    // Final limit: ISO 6346 containers are max 11 characters
    if (result.length() > 11) {
        result = result.substr(0, 11);
        LOGD("Truncated to 11 characters: '%s'", result.c_str());
    }
    
    LOGD("Final horizontal container text: '%s'", result.c_str());
    return result;
}

std::string YoloOCREngine::assembleText(
    std::vector<CharBox>& boxes,
    const std::string& classType
) {
    if (classType == "code_container_v") {
        // Vertical container: sort by Y coordinate (top to bottom)
        std::sort(boxes.begin(), boxes.end(),
            [](const CharBox& a, const CharBox& b) { return a.y < b.y; });
        LOGD("Sorting characters vertically (top to bottom) for %s", classType.c_str());
        
        // For containers: limit to 11 characters (ISO 6346)
        if (boxes.size() > 11) {
            boxes.resize(11);
        }
        
        // Build text
        std::string text;
        for (const auto& box : boxes) {
            text += box.character;
        }
        
        LOGD("Assembled text: %s", text.c_str());
        return text;
        
    } else if (classType == "code_container_h") {
        // Horizontal container: multi-line assembly with greedy top line
        return assembleHorizontalContainerText(boxes);
        
    } else {
        // Other types: simple left-to-right sorting
        std::sort(boxes.begin(), boxes.end(),
            [](const CharBox& a, const CharBox& b) { return a.x < b.x; });
        LOGD("Sorting characters horizontally (left to right) for %s", classType.c_str());
        
        // Build text
        std::string text;
        for (const auto& box : boxes) {
            text += box.character;
        }
        
        LOGD("Assembled text: %s", text.c_str());
        return text;
    }
}

float YoloOCREngine::calculateConfidence(const std::vector<CharBox>& boxes) {
    if (boxes.empty()) return 0.0f;
    
    // Average confidence of all detected characters
    float sum = std::accumulate(boxes.begin(), boxes.end(), 0.0f,
        [](float acc, const CharBox& box) { return acc + box.confidence; });
    
    return sum / boxes.size();
}

} // namespace universalscanner