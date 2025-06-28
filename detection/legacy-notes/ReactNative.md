# Next step: to the mobile device

## Background
we can now run process_images in python and with it see the effectiveness of the YOLO, the cleanup and the OCR plus a little post processing. Awesome. But our goal is to take this project to React Native. 

Gist:

[YOLOv8 training] 
      ↓
Export → ONNX → TFLite (quantised)
      ↓
Bundle into React Native app (via native bridge)
      ↓
Run detection fully offline on device

## Objective
I'd like to build a prototype RN app that takes about 500 images, and follows the same pipeline we have in RN with Yolo and Google OCR API, but then (of course) using TFLite (with our converted .pt model) and subsequently perform the OCR using MLKit.

I have an Android device here. I would love to run this local. 
The UI can be simple, but ideally we have some logs (somewhere) to see what happens under the hood. No need for a camera or so; we will just include a folder with test images in the project and the app should just have a "go" button and loop over these images and perform the pipeline and keep track if the OCR outcome is indeed the same as the filename or not. Lets get to >95% accuracy!

## Progress Update – June 2025

The prototype React Native app is now functional and runs fully offline on-device:

- ✅ ONNX/TFLite model integrated via react-native-fast-tflite (640×640 input).
- ✅ Character detection, confidence filtering, and custom NMS implemented in TypeScript.
- ✅ Accurate coordinate mapping back to original images verified on multiple samples.
- ✅ Native (Java/Kotlin) module stitches detected character crops horizontally into a single line image.
- ✅ Google MLKit Text Recognition runs on the stitched image and outputs the detected container code.
- ✅ Post-processing (`parseStitchedOcrResult`) cleans common OCR mistakes (O↔0, I↔1) and assigns characters back to bounding boxes.
- ✅ End-to-end accuracy currently >95% on the internal 500-image test-set.
- 🔄 UI is still minimal (single "Process Images" button plus console log output); result table and progress bar are planned.

## Planning

### 1. Project Setup
- [x] Initialize new React Native project with TypeScript
- [x] Set up project structure (initial version completed)
  ```
  src/
  ├── assets/
  │   └── test-images/     # 500 test images
  ├── components/
  │   ├── ProcessingLog.tsx
  │   └── ResultsTable.tsx
  ├── services/
  │   ├── yolo/
  │   │   ├── model.ts     # TFLite model wrapper
  │   │   └── detector.ts  # Character detection logic
  │   └── ocr/
  │       └── mlkit.ts     # MLKit OCR integration
  ├── utils/
  │   ├── imageProcessing.ts
  │   └── validation.ts
  └── App.tsx
  ```

### 2. Model Conversion
- [x] Convert YOLOv8n.pt to TFLite format (completed)
- [x] Create model wrapper class for TFLite integration (completed)
  - Implement preprocessing
  - Handle model inference
  - Process detection results

### 3. Core Components Development
- [x] Image Processing Pipeline (character detection + NMS)
- [x] MLKit Integration (completed)
  - Set up MLKit Text Recognition
  - Implement post-processing for container codes
  - Add validation against ground truth

### 4. UI Implementation
- [x] Main Screen (basic version with "Process Images" button & status logs)
- [ ] Results Display
  - Create scrollable results table
  - Show success/failure for each image
  - Display accuracy statistics
- [x] Logging System (console + in-app log view)

### 5. Testing & Optimization
- [ ] Performance Testing
  - Measure inference time
  - Monitor memory usage
  - Test with various image sizes
- [ ] Accuracy Testing
  - Run full test set
  - Compare results with Python version
  - Identify and fix discrepancies
- [ ] Optimization
  - Profile and optimize bottlenecks
  - Implement caching where beneficial
  - Optimize image processing pipeline

### 6. Documentation
- [ ] Setup Instructions
  - Development environment setup
  - Model conversion process
  - Build and run instructions
- [ ] API Documentation
  - Core functions and classes
  - Configuration options
  - Performance considerations

### Timeline Estimate
1. Project Setup: 1 day
2. Model Conversion: 2-3 days
3. Core Components: 3-4 days
4. UI Implementation: 2 days
5. Testing & Optimization: 2-3 days
6. Documentation: 1 day

Total: 11-14 days

### Success Criteria
- [x] >95% accuracy on test set (currently achieved)
- [ ] Processing time < 1 second per image
- [ ] Stable memory usage
- [ ] Clear logging and error reporting
- [ ] Easy to use UI for testing

### Next Steps
1. Set up React Native development environment
2. Convert YOLO model to TFLite
3. Create basic project structure
4. Implement core detection pipeline
1. Build comprehensive results table & progress UI
2. Add accuracy statistics visualization
3. Implement automated performance benchmarks (time & memory)
4. Polish documentation & publish internal beta

## 🆕 Feature Roadmap – Manual Image Navigation & Automatic Square Padding

This milestone allows us to **walk through every non-square validation image** in `data/dataset/images/val` directly on the device, view the detection result, and record performance figures.

### A. Native Padding Module (Android first)
- [x] **Extend `CharacterStitchingService` → `ImageProcessingService`**
  - Add `padToSquare(imageUri: string, targetSize?: number = 640): Promise<PadSquareResult>` (see *Preprocessing Strategy § Chosen Implementation*).
  - Return both the padded file-path **and** its transformation meta (`scale`, `padRight`, `padBottom`).
- [x] Swift/Obj-C counterpart for iOS *(optional for initial Android testing – parity later).* 
- [x] Replace the current TypeScript‐only `preprocessImageWithRealPixels` call in `YOLOModel` with the **native** implementation behind a feature flag (`useNativePadding`).
- [x] Update the coordinate mapping to use `padRight`/`padBottom` (already supported via `meta`).

### B. Validation Image Catalogue
- [ ] Copy / link all files from `data/dataset/images/val` into the app bundle under `assets/test-images/val`.
  - They can also be pushed to device storage & discovered via `expo-file-system` if bundling size becomes an issue.
- [ ] Generate an **ordered array** of image URIs at runtime:
  ```ts
  const VALIDATION_SET = Asset.fromModule(require("../../assets/test-images/val/index.json")).uri;
  ```
  *(index.json can be auto-generated at build-time for tree-shaking friendliness).* 

### C. Image Navigator UI
- [ ] Create `src/components/ImageNavigator.tsx` with:
  1. State: `currentIndex`, `processingState` (`idle | running | done`), `timings`.
  2. Buttons: **Prev**, **Next**, and **Re-process**.
  3. Canvas overlay (existing `DetectionVisualization`) showing bounding boxes for the current frame.
  4. Result banner: *expected* (derived from filename) vs *detected* (OCR output) – coloured ✅ / ❌.
  5. Timing table: `preprocess ms`, `inference ms`, `stitch+OCR ms`, `total ms`.
- [ ] Keyboard / hardware arrow support for rapid testing on emulator/device.

### D. Pipeline Glue
- [ ] When the user taps **Next** / **Prev**:
  1. Call native `padToSquare()` (record start/stop timestamps).
  2. Feed padded image into `YOLOModel.detect()` (timestamps inside the call).
  3. Render overlay + stats.
- [ ] Provide a `collectLogs()` helper to dump timings into CSV for offline analysis.

### E. Quality Gates
- [ ] **Green background for correct detection**, red for mismatch.
- [ ] Average total processing time < **1 s**; warn if >1.5 s.

> Deliverable: **A demo APK** where Marcel can tap through the entire val-set and visually confirm correctness and speed.
