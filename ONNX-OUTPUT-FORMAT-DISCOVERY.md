# ONNX Output Format Discovery

**Date**: 2025-06-27  
**Critical Discovery**: ONNX Runtime React Native returns nested arrays, NOT flat Float32Array

## The Problem

We experienced extremely low confidence scores (0.01-0.05) and garbage bounding box data when processing ONNX model outputs. The issue was caused by incorrect data access patterns.

## Root Cause

ONNX Runtime for React Native returns outputs in a **nested array format**, not the expected flat `Float32Array` that desktop/server versions use.

## Wrong Approach (Causes Garbage Data)

```typescript
// ❌ WRONG - This treats nested arrays as flat data
const outputData = output.data as Float32Array;
const centerX = outputData[i];
const centerY = outputData[i + 1]; 
const width = outputData[i + 2];
const height = outputData[i + 3];
```

## Correct Approach

```typescript
// ✅ CORRECT - Proper nested array handling
const raw3d = output.value as number[][][];
const preds2d = raw3d[0]; // Extract 2D array from 3D wrapper

// Handle both possible orientations: [ATTRIBUTES, N] or [N, ATTRIBUTES]
const attributes = 9; // 4 bbox + 5 classes for YOLOv7 model
const predsAlongLastDim = preds2d[0].length !== attributes;

function getVal(anchorIdx: number, featureIdx: number): number {
  return predsAlongLastDim 
    ? preds2d[featureIdx][anchorIdx]  // Format: [ATTRIBUTES, N]
    : preds2d[anchorIdx][featureIdx]; // Format: [N, ATTRIBUTES]
}

// Now access data correctly
const centerX = getVal(i, 0);
const centerY = getVal(i, 1);
const width = getVal(i, 2);
const height = getVal(i, 3);
const confidence = getVal(i, 4);
```

## Model Output Shape Analysis

For our YOLOv7 container detection model:
- **Expected Shape**: `[1, 9, 8400]` or `[1, 8400, 9]`
- **Attributes**: 9 (4 bbox + 5 classes)
- **Anchors**: 8400 detection points

The key insight is that the first dimension (batch size = 1) is preserved, but the data structure is a **3D nested array**, not a flat buffer.

## Implementation Reference

See working implementation in:
`/ContainerCameraApp/android/app/src/main/java/com/cargosnap/app/YoloBridgeModule.kt`

## Impact

This discovery fixed:
- ❌ Confidence scores jumping from 0.01 → ✅ 0.84+
- ❌ Random bounding boxes → ✅ Accurate object detection
- ❌ Model appearing "broken" → ✅ Proper YOLO inference

## Key Takeaway

**ONNX Runtime React Native != ONNX Runtime Desktop/Server**

Always check the actual data structure returned by `output.value` rather than assuming it matches documentation for other platforms.