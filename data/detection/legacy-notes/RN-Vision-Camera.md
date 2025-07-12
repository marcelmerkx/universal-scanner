# Implement YOLO-OCR on RN Vision camera

## technical planning

Below is a complete, implementation-ready roadmap for the "RN-Vision-Camera ↔ YOLO-OCR" feature.
It is split into (1) end-to-end flow, (2) detailed work packages with sequencing, (3) key architectural/technical decisions, and (4) optional shortcuts or enablers that could accelerate delivery.

### 1 · End-to-End Flow (desired runtime behaviour)

1. User taps the "Scan container" tab → we immediately start a "warm-up" of the VisionCamera to shorten TTF (time-to-first-frame). When the user presses the round shutter button:
    * Stopwatch starts.
    * VisionCamera streams frames to a Frame Processor (JSI/TurboModule).

The Frame Processor:

    i. Crops the middle vertical guide overlay area (keeps UX clear, reduces compute).
    ii. Performs letter-box padding to 640 × 640 px.
    iii. Converts to Float32 BCHW tensor (0–1 normalised).
    iv. Runs ONNX Runtime Mobile "best-ocr.onnx" on CPU (or NNAPI delegate when available).
    Runs post-processing (threshold, NMS, TOP_K = 11, y-sort).
    vi. Emits an array of 11 detections ←→ JS bridge.

JS concatenates the 11 characters to one string → renders overlay boxes and the decoded code.
Stopwatch stops; number of model iterations is the frame count; both metrics are shown under the result.

Camera view remains open with shutter still visible for a subsequent scan.
Analytics event "scan_complete" with timing & success flag is logged.

### 2 · Work Packages & Sequencing

#### WP-0 Infrastructure
Bump React Native ≥ 0.73 (for stable JSI & Hermes).
Add react-native-vision-camera v3, react-native-onnxruntime, react-native-fs, and Reanimated v3.

#### WP-1 Frame Processor plug-in
Create FrameProcessorPlugin_YoloOcr.cpp/.h/.ts (VisionCamera ≥ 3 provides the template).
Implement JNI/Obj-C++ glue to call ONNX Runtime Mobile synchronously inside the frame thread.
Expose simple JS API: detectCharacters(frame, options) → Detection[].

#### WP-2 Model packaging
Keep ONNX. But in a later phase we may want to evaluate ORT. 

> **Why ORT?**  The ONNX Runtime can load plain *.onnx* files, _but_ exporting the graph to the proprietary **ORT format** (``onnxruntime_tools.convert_onnx_models_to_ort``) strips unused weights and serialises operator kernels.  This lowers model-load latency (~30 %) and disk size (~15 %).  If the optimisation step is skipped the scanner will still work—use the untouched model located at `models/best-OCR-14-06-25.onnx`.

Copy the resulting file into `android/app/src/main/assets/`.
(When we later add iOS support the same file goes into the iOS bundle.)

#### WP-3 UI & State machine
Screen with Camera view, centre overlay, shutter.
Result state with bounding boxes, code string, time-to-result, iteration count.
"Scan again" resets UI and restarts frame stream.

#### WP-4 Performance safeguards
Implement frame-stride (e.g. analyse every 2nd frame on low-end) and a 4 s detection timeout. Monitor memory pressure via `onTrimMemory` (Android).

#### WP-5 Release & Documentation
Update README and RN-Vision-Camera.md with a pipeline diagram, troubleshooting, and known-issue FAQ.
Produce a short Loom walkthrough.
Indicative timeline (1 FTE mobile + 0.3 FTE CV): 10 calendar days.

### 3 · Key Technical Decisions (finalised)

#### A. Inference engine → ONNX Runtime Mobile (locked-in)
Plain *.onnx* or ORT format as explained above. No TFLite path for v1.

#### B. Frame Processor language → Kotlin (Android-only first)
JNI wrapper around ORT; C++ cross-platform refactor is deferred until iOS work starts.

#### C. Post-processing location → TypeScript (JS)
Maintainability wins; can migrate to C++ once we profile bottlenecks.

#### D. Overlay rendering → react-native-svg
Simple, mature, adequate performance; upgrade to Reanimated-driven overlays later if required.

#### E. Timing metric collection → dual
Capture timestamps both inside the Frame Processor (native, precise) and in JS (UI level) to cross-check end-to-end latency.

### 4 · Shortcuts / Enablers

1. Prototype with **JS-only `useFrameProcessor`**; port to Kotlin after baseline FPS and accuracy look good.
2. ~~Re-use Ultralytics' pre-built YOLOv8-digit-letter model if acceptable licensing.~~ _Not needed—our current model performs adequately._
3. Skip iOS for the initial release; focus all effort on Android.
4. ~~Use Expo Camera + WebAssembly ORT…~~ VisionCamera remains the default camera stack.
5. **Evaluate `react-native-vision-camera-ocr-scanner`**
   *Pros*
   • Ready-made Kotlin/Swift frame processor skeleton  
   • Handles buffer <→ bitmap conversion and thread scheduling  
   • MIT licence, active maintenance  
   *Cons*
   • Adds native dependency surface; may diverge from latest VisionCamera API  
   • Still needs custom post-processing code for 11-character constraint  
   • Extra abstraction could complicate future C++ migration  
   *Implications*
   • Might shave 1–2 development days at the cost of another third-party layer.  
   • Decide after JS prototype—if frame dispatch code becomes a time sink we adopt it; otherwise keep lean in-house plugin.

Outcome
The above plan delivers an on-device, single-pass YOLO-OCR scanner with clear milestones, risk-reducing toggles, and measurable success KPIs (TTF < 0.4 s, ≥97 % accuracy).
