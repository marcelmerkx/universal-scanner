# ONNX C++ Implementation Guide for Universal Scanner

**Date**: 2025-01-03  
**Critical**: This document captures hard-won knowledge about ONNX output format in C++ implementation

## 🚨 Key Discovery: ONNX Output Format in C++

After hours of debugging, we discovered that the ONNX model output has a specific structure that must be handled correctly to achieve proper detection confidence.

## Output Tensor Structure

### Model: unified-detection-v7.onnx
- **Input Shape**: `[1, 3, 640, 640]` (NCHW format)
- **Output Shape**: `[1, 9, 8400]`
  - Batch size: 1
  - Features: 9 (4 bbox + 1 objectness + 4 classes)
  - Anchors: 8400 detection points

### Feature Layout (9 features per anchor)
```
Index 0-3: Bounding box (x_center, y_center, width, height)
Index 4:   Objectness score (raw logit, needs sigmoid)
Index 5-8: Class scores for 4 classes (raw logits, need sigmoid)
           - Class 0: code_qr_barcode
           - Class 1: license_plate
           - Class 2: code_container_h
           - Class 3: code_container_v
```

## Critical Implementation Details

### 1. Tensor Format Flexibility

The ONNX output can be in two possible formats:
- **Features-major**: `[1, 9, 8400]` - features × anchors
- **Anchors-major**: `[1, 8400, 9]` - anchors × features

Always implement adaptive indexing:

```cpp
// Determine tensor format
bool isFeaturesMajor = true;
float sample_obj_fm = output[4 * anchors + 100];  // features-major
float sample_obj_am = output[100 * features + 4]; // anchors-major

// YOLO objectness logits should be in range ~[-10, 10]
if (std::abs(sample_obj_fm) > 15.0f && std::abs(sample_obj_am) < 15.0f) {
    isFeaturesMajor = false;
}

// Adaptive indexing helper
auto getVal = [&](size_t anchorIdx, size_t featureIdx) -> float {
    if (isFeaturesMajor) {
        return output[featureIdx * anchors + anchorIdx];
    } else {
        return output[anchorIdx * features + featureIdx];
    }
};
```

### 2. Sigmoid Activation Required

**CRITICAL**: All objectness and class scores are raw logits that MUST have sigmoid applied:

```cpp
auto sigmoid = [](float x) {
    return 1.0f / (1.0f + std::exp(-x));
};

float objectness = sigmoid(getVal(anchor, 4));
float classScore = sigmoid(getVal(anchor, 5 + classIdx));
```

### 3. Confidence Calculation

**IMPORTANT DISCOVERY**: This model outputs very low objectness scores (~0.01 raw, ~0.50 after sigmoid).

Traditional YOLO confidence calculation:
```cpp
float confidence = objectness * classScore;  // Gives ~33% for our model
```

For this specific model, use class score only:
```cpp
float confidence = classScore;  // Gives ~67% (correct)
```

### 4. Complete Detection Pipeline

```cpp
for (size_t a = 0; a < anchors; a++) {
    // Get bbox coordinates (already in correct scale)
    float x_center = getVal(a, 0);
    float y_center = getVal(a, 1);
    float width    = getVal(a, 2);
    float height   = getVal(a, 3);
    
    // Get objectness (not used for confidence in this model)
    float objectness_raw = getVal(a, 4);
    float objectness = sigmoid(objectness_raw);
    
    // Find best class
    float maxClassProb = 0.0f;
    int classIdx = -1;
    for (int c = 0; c < 4; c++) {
        float classProb_raw = getVal(a, 5 + c);
        float classProb = sigmoid(classProb_raw);
        if (classProb > maxClassProb) {
            maxClassProb = classProb;
            classIdx = c;
        }
    }
    
    // Use class probability as confidence
    float confidence = maxClassProb;
    
    // Filter detections
    if (confidence > 0.5f) {
        // Process detection
    }
}
```

## Common Pitfalls to Avoid

1. **❌ Don't assume flat Float32Array format** - ONNX C++ API returns a pointer to float data
2. **❌ Don't forget sigmoid activation** - Raw logits will give wrong results
3. **❌ Don't hardcode tensor layout** - Always use adaptive indexing
4. **❌ Don't assume traditional YOLO confidence** - This model has low objectness scores

## Debugging Tips

1. **Check tensor values**: Early anchors often have unrealistic values (>20), real detections are usually in the middle range (anchors 6900-7000 for license plates)

2. **Verify preprocessing**: 
   - Input should be normalized to [0,1]
   - Apply proper rotation (90° CCW for landscape frames)
   - Use white padding to 640×640

3. **Log key metrics**:
   ```cpp
   LOGF("Detection stats: >10%%=%d, >25%%=%d, >50%%=%d, >70%%=%d", 
        candidatesAbove10, candidatesAbove25, candidatesAbove50, candidatesAbove70);
   ```

## Model-Specific Notes

This `unified-detection-v7.onnx` model:
- Trained specifically for logistics/transportation objects
- Uses non-standard objectness scoring (very low values)
- Best results with class-probability-only confidence
- License plates typically detected around anchors 6930-6936
- Expected confidence for well-visible license plates: 65-70%

## References

- Original investigation: `/Users/marcelmerkx/Development/universal-scanner/ONNX-OUTPUT-FORMAT-DISCOVERY.md`
- Working implementation: `/Users/marcelmerkx/Development/universal-scanner/android/src/main/cpp/Universal.cpp`
- Model architecture: YOLOv8n-based with custom training