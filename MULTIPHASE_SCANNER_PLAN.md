

### Code Types to Support
- `code_qr_barcode` — Linear barcodes (EAN, Code128) and 2D QR codes (supported using MLKit)
- `code_license_plate` — Alphanumeric plates (generic)
- `code_container_h` — ISO 6346 container codes (horizontal)
- `code_container_v` — ISO 6346 container codes (vertical)
- `code_seal` — Security seals with serials (often dual: QR + text)

Count on adding in PHASE 2:
- `text_printed` — Freeform printed text (OCR)
- `code_lcd_display` — Casio/LCD-style digit panels
- `code_rail_wagon` — Railcar identifier codes
- `code_container_airfreight` — ULD identifiers for airfreight
- `code_vin` — Vehicle Identification Numbers (ISO 3779, often dual: QR + text)




# ONNX Output Format Discovery - Critical Finding

**Date**: 2025-06-27  
**Context**: Mobile app v7 model integration debugging  
**Issue**: Mobile app getting 6% max confidence vs Python 97% accuracy  

## 🚨 Root Cause Discovered

The fundamental issue was a **complete misunderstanding of ONNX Runtime React Native output format**.

### ❌ What We Were Doing Wrong

```typescript
// INCORRECT - treating output as flat Float32Array
const outputData = output.data as Float32Array;
const centerX = outputData[i];
const centerY = outputData[numPredictions + i];
```

### ✅ What The Working Implementation Does

```kotlin
// CORRECT - output is nested array structure
val raw3d = out[0].value as Array<Array<FloatArray>>
val (code, _) = postprocess(raw3d[0])  // Pass 2D slice
```

## 📊 The Real ONNX Output Structure

### For Character Recognition (Legacy)
- **Shape**: `[1, 40, 8400]`
- **Format**: `Array<Array<FloatArray>>`
- **Access**: `output.value[0]` gives `Array<FloatArray>` of shape `[40, 8400]`

### For Object Detection (Current v7 Model)
- **Shape**: `[1, 9, 8400]` 
- **Format**: `Array<Array<FloatArray>>`
- **Access**: `output.value[0]` gives `Array<FloatArray>` of shape `[9, 8400]`
- **Attributes**: 4 bbox coords + 5 class scores = 9 total

## 🔄 Shape Adaptability

The working implementation handles **two possible orientations**:

```kotlin
val predsAlongLastDim = preds[0].size != attributes
fun getVal(anchorIdx: Int, featureIdx: Int): Float {
  return if (predsAlongLastDim) {
    preds[featureIdx][anchorIdx]      // [ATTRIBUTES, N] format
  } else {
    preds[anchorIdx][featureIdx]      // [N, ATTRIBUTES] format  
  }
}
```

This means the model output can be either:
- `[9, 8400]` - attributes first (9 features × 8400 anchors)
- `[8400, 9]` - anchors first (8400 anchors × 9 features)

## 🎯 Key Differences: Character vs Object Detection

| Aspect | Character Recognition | Object Detection |
|--------|----------------------|------------------|
| **Classes** | 36 (0-9, A-Z) | 5 (container_h, container_v, etc.) |
| **Attributes** | 4 + 36 = 40 | 4 + 5 = 9 |
| **Output Shape** | `[1, 40, 8400]` | `[1, 9, 8400]` |
| **Confidence Threshold** | 0.25 | Should be ~0.5 |
| **Post-processing** | Take top 11, sort by Y | NMS + confidence filter |

## 🔍 Evidence From Logs

Our incorrect implementation was producing:
```
Max confidence found: 0.0610 (6.1%)
Found 8355 raw detections before NMS
After NMS: 1749 detections  
Final detections after confidence filter (0.1): 0
```

This suggests we were reading **garbage data** due to incorrect array indexing, not actual low-confidence predictions.

## 🛠️ Required Fix

1. **Change output parsing**:
   ```typescript
   // FROM: output.data as Float32Array
   // TO: output.value as number[][][]
   const raw3d = output.value as number[][][];
   const preds2d = raw3d[0]; // Shape: [9, 8400] or [8400, 9]
   ```

2. **Implement shape detection**:
   ```typescript
   const attributes = 9; // 4 bbox + 5 classes for v7 model
   const predsAlongLastDim = preds2d[0].length !== attributes;
   ```

3. **Use adaptive indexing**:
   ```typescript
   function getVal(anchorIdx: number, featureIdx: number): number {
     return predsAlongLastDim 
       ? preds2d[featureIdx][anchorIdx]  // [9, 8400]
       : preds2d[anchorIdx][featureIdx]; // [8400, 9]
   }
   ```

## 📚 Reference Implementation

See: `/ContainerCameraApp/android/app/src/main/java/com/cargosnap/app/YoloBridgeModule.kt`
- Lines 90-96: Correct output extraction
- Lines 152-211: Proper postprocessing with shape detection
- Lines 159-165: Adaptive indexing implementation

## 🎯 Next Steps

1. ✅ Document this finding (this file)
2. ✅ Update project memory 
3. 🔄 Restore confidence threshold to >0.5 (>50%)
4. 🔄 Implement correct ONNX output parsing
5. 🔄 Test with corrected implementation

---

**Critical Lesson**: Always verify the actual data structure returned by external libraries, especially when porting between platforms or frameworks. The difference between `output.data` and `output.value` was the key to unlocking proper model performance.