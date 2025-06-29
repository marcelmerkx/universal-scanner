# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a universal scanner mobile app built with React Native that detects and decodes multiple code types offline and in real-time. The system is designed as a `react-native-vision-camera` plugin with a standalone demo app.

**Core Architecture (Hybrid MLKit + ONNX):**
- UI Layer: React Native (TypeScript) + `react-native-vision-camera`
- Native Layer: Shared C++ core (JNI for Android, Obj-C++ for iOS)
- Detection: YOLOv8n model (ONNX Runtime)
- Recognition: MLKit (via JNI/Obj-C++ bridges) for proven reliability, with fallback to ONNX Runtime OCR models and Tesseract for complex text
- Domain-Specific Pipelines: Code-type specific processing and validation

## Supported Code Types

The scanner supports 11 different code types with specific class names:
- `code_qr_barcode` — 2D QR codes
- `code_license_plate` — Alphanumeric plates (generic)
- `code_container_h` — ISO 6346 container codes (horizontal)
- `code_container_v` — ISO 6346 container codes (vertical)
- `text_printed` — Freeform printed text (OCR)
- `code_seal` — Security seals with serials (often dual: QR + text)
- `code_lcd_display` — Casio/LCD-style digit panels
- `code_rail_wagon` — Railcar identifier codes
- `code_air_container` — ULD identifiers for airfreight
- `code_vin` — Vehicle Identification Numbers (ISO 3779, often dual: QR + text)

## Key DTOs

**Input Configuration:**
```typescript
interface ScannerConfig {
  enabledTypes: string[];
  regexPerType?: Record<string, string[]>;
  manualMode?: boolean;
  verbose?: boolean;
}
```

**Output Result:**
```typescript
interface ScanResult {
  type: string;
  value: string;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  imageCropPath: string;
  fullFramePath: string;
  model: string;
  verbose?: Record<string, any>;
}
```

## Development Approach

- **Stay real**: use the camera. we will not use mock data or placeholder values or generated images unless explicitly agreed to.
- **Key and parameters**: never use or apply API keys or static project ID's. Ask me to put them in the .env file. Really.
- **Plugin-First Architecture**: Build as a reusable `react-native-vision-camera` frame processor plugin
- **Shared Native Core**: C++ implementation to avoid Android/iOS logic duplication
- **Hybrid ML Strategy**: MLKit for proven text/barcode recognition with ONNX Runtime for YOLO detection
- **Domain-Specific Pipelines**: Specialized processing for each code type (container validation, license plate formatting, etc.)
- **Offline-First**: All inference runs on-device with optimized models
- **Performance Optimized**: Native-heavy processing to minimize RN bridge usage
- **Verbose Mode**: Rich debugging output for development and QA
- **Manual Targeting**: User can tap specific codes in cluttered scenes

## Project Status

- **Baseline Established**: Forked react-native-fast-tflite as proven VisionCamera + JSI foundation
- **Documentation Recovered**: All user stories, technical architecture, and training assets restored
- **TensorFlow Lite Demo Working**: Example app builds and runs successfully on Android
- **Detection Assets Ready**: YOLOv7 models and training scripts available in `/detection/`
- **Current Focus**: Phase 3 - TensorFlow Lite → ONNX Runtime migration

## Development Workflow

- Upon completion of each project step, update the project_plan with the completions and the status outcomes and propose a git commit step
- Use TypeScript for all React Native code with strict type checking
- Follow existing code patterns and MLKit integration examples from README.md
- Prioritize performance by keeping heavy processing in native C++ layer
- Test on both Android and iOS platforms (Android can be tested with emulator, iOS requires full Xcode)

## Key Technical Patterns

**CRITICAL: ONNX Runtime React Native Output Format**
⚠️ **MAJOR DISCOVERY (2025-06-27)**: ONNX-RN returns nested arrays, NOT flat Float32Array!

```typescript
// ❌ WRONG - causes garbage data and low confidence
const outputData = output.data as Float32Array;
const centerX = outputData[i];

// ✅ CORRECT - proper nested array handling  
const raw3d = output.value as number[][][];
const preds2d = raw3d[0]; // Shape: [9, 8400] or [8400, 9]

// Handle both orientations with adaptive indexing
const attributes = 9; // 4 bbox + 5 classes for v7 model
const predsAlongLastDim = preds2d[0].length !== attributes;
function getVal(anchorIdx: number, featureIdx: number): number {
  return predsAlongLastDim 
    ? preds2d[featureIdx][anchorIdx]  // [ATTRIBUTES, N]
    : preds2d[anchorIdx][featureIdx]; // [N, ATTRIBUTES]
}
```

Reference: Working implementation needed (to be created during ONNX migration)
Documentation: `/ONNX-OUTPUT-FORMAT-DISCOVERY.md`

**MLKit Integration via C++:**
```cpp
// Android: JNI bridge to MLKit
class MLKitRecognizer : public ITextRecognizer {
    jobject mlkitManager;
    std::string recognizeText(const cv::Mat& image) override;
};

// iOS: Obj-C++ bridge to MLKit
@interface MLKitBridge : NSObject
- (NSString*)recognizeTextFromImage:(UIImage*)image;
@end
```

**Domain-Specific Processing:**
```cpp
class ContainerProcessor : public ICodeProcessor {
    bool validateISO6346(const std::string& code);
    std::string normalizeFormat(const std::string& raw);
};
```

## Current Phase: TensorFlow Lite → ONNX Runtime Migration

**Next Steps**:
1. Test current TensorFlow Lite example app on device
2. Replace TensorFlow Lite with ONNX Runtime in C++ core
3. Integrate YOLOv7 detection models from `/detection/models/`
4. Add MLKit integration alongside ONNX Runtime
5. Implement Universal Scanner plugin architecture

**Available Assets**:
- `/detection/models/unified-detection-v7.onnx` - Trained YOLOv7 model
- `/detection/scripts/` - Training and testing scripts
- `/detection/notebooks/` - Model training notebooks
- `/ONNX-OUTPUT-FORMAT-DISCOVERY.md` - Critical output format handling

**Migration Strategy**:
- Keep existing VisionCamera Frame Processor integration
- Replace TensorFlow Lite C++ code with ONNX Runtime
- Maintain JSI bindings and native module structure
- Add Universal Scanner specific features incrementally

## VisionCamera Frame Processor Architecture

**Performance Constraints**:
- **Frame Time Limits**: 30 FPS = 33ms, 60 FPS = 16ms per frame
- **Memory Intensive**: 4K frames = ~12MB each, 60 FPS = ~700MB/second
- **Zero-Copy Pattern**: Use direct GPU buffer access, avoid frame copying
- **JSI Performance**: Native plugin calls add only ~1ms overhead

**Frame Processor Plugin Patterns**:
```typescript
// Worklet-based frame processing
const frameProcessor = useFrameProcessor((frame) => {
  'worklet'
  
  // Call native plugin with direct frame access
  const results = universalScanner(frame, {
    enabledTypes: ['container', 'qr', 'barcode'],
    confidence: 0.7,
    maxResults: 5
  })
  
  // Handle results on JS thread
  if (results) {
    runOnJS(handleScanResults)(results)
  }
}, [])
```

**C++ Plugin Implementation Strategy**:
```cpp
// Plugin signature following VisionCamera patterns
static facebook::jsi::Value universalScanner(
    facebook::jsi::Runtime& runtime,
    const facebook::jsi::Value& thisValue,
    const facebook::jsi::Value* arguments,
    size_t count
) {
    // 1. Extract frame and config
    auto frame = arguments[0].asObject(runtime);
    auto config = arguments[1].asObject(runtime);
    
    // 2. Process with ONNX + MLKit pipeline
    auto results = processingEngine.scan(frame, config);
    
    // 3. Return structured results
    return convertToJSI(runtime, results);
}
```

**Threading and Async Processing**:
- **Synchronous Path**: Quick processing within frame time budget
- **Async Path**: Copy frame, dispatch to background thread, emit results via events
- **Queue Management**: Handle frame dropping for sustained performance

**Memory Management**:
- Reuse allocated buffers for tensor processing
- Minimize heap allocations during frame processing
- Use memory pools for frequent allocations
- Proper cleanup of GPU resources

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.