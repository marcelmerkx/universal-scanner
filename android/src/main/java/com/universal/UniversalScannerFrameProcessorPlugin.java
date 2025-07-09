package com.universal;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import android.util.Log;

import com.mrousavy.camera.frameprocessors.Frame;
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin;
import com.mrousavy.camera.frameprocessors.VisionCameraProxy;
import com.mrousavy.camera.core.FrameInvalidError;

import org.json.JSONObject;
import org.json.JSONArray;

import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;

public class UniversalScannerFrameProcessorPlugin extends FrameProcessorPlugin {
    private static final String TAG = "UniversalScanner";
    private UniversalNativeModule nativeModule;
    private static android.content.Context appContext = null;
    private static boolean modelExtracted = false;
    
    // Static method to set the application context
    public static void setApplicationContext(android.content.Context context) {
        appContext = context;
    }

    public UniversalScannerFrameProcessorPlugin(@NonNull VisionCameraProxy proxy) {
        super();
        try {
            // Extract model to internal storage if not already done
            if (!modelExtracted && appContext != null) {
                extractModelFromAssets();
            }
            
            nativeModule = new UniversalNativeModule();
            Log.i(TAG, "Native module initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize native module", e);
            nativeModule = null;
        }
    }
    
    private void extractModelFromAssets() {
        try {
            // Extract ONNX models only (TFLite not needed for now)
            extractOnnxModels();
            
            modelExtracted = true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to extract models from assets", e);
        }
    }
    
    private void extractOnnxModels() {
        try {
            // Try to extract NNAPI-compatible model first
            java.io.InputStream is = null;
            java.io.File outputFile = null;
            String modelName = null;
            
            try {
                // Try NNAPI-compatible model first
                is = appContext.getAssets().open("unified-detection-v7-nnapi.onnx");
                outputFile = new java.io.File(appContext.getFilesDir(), "unified-detection-v7-nnapi.onnx");
                modelName = "unified-detection-v7-nnapi.onnx";
                Log.i(TAG, "Found NNAPI-compatible ONNX model in assets");
            } catch (Exception e) {
                Log.w(TAG, "NNAPI ONNX model not found, falling back to original model");
                try {
                    // Fallback to original model
                    is = appContext.getAssets().open("unified-detection-v7.onnx");
                    outputFile = new java.io.File(appContext.getFilesDir(), "unified-detection-v7.onnx");
                    modelName = "unified-detection-v7.onnx";
                } catch (Exception e2) {
                    Log.e(TAG, "No ONNX model found in assets", e2);
                    return;
                }
            }
            
            copyStreamToFile(is, outputFile);
            Log.i(TAG, "ONNX detection model extracted to: " + outputFile.getAbsolutePath());
            
            // Extract OCR model
            try {
                java.io.InputStream ocrIs = appContext.getAssets().open("best-OCR-Colab-22-06-25.onnx");
                java.io.File ocrOutputFile = new java.io.File(appContext.getFilesDir(), "best-OCR-Colab-22-06-25.onnx");
                copyStreamToFile(ocrIs, ocrOutputFile);
                Log.i(TAG, "OCR model extracted to: " + ocrOutputFile.getAbsolutePath());
            } catch (Exception ocrE) {
                Log.w(TAG, "OCR model not found in assets", ocrE);
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to extract ONNX models from assets", e);
        }
    }
    
    private void extractTfliteModels() {
        // Extract TFLite models for A/B testing
        String[] tfliteModels = {
            "unified-detection-v7_int8.tflite",
            "unified-detection-v7_float16.tflite"
        };
        
        for (String modelName : tfliteModels) {
            try {
                java.io.InputStream is = appContext.getAssets().open(modelName);
                java.io.File outputFile = new java.io.File(appContext.getFilesDir(), modelName);
                copyStreamToFile(is, outputFile);
                Log.i(TAG, "TFLite model extracted: " + outputFile.getAbsolutePath());
            } catch (Exception e) {
                Log.w(TAG, "TFLite model not found: " + modelName);
            }
        }
    }
    
    private void copyStreamToFile(java.io.InputStream is, java.io.File outputFile) throws Exception {
        java.io.OutputStream os = new java.io.FileOutputStream(outputFile);
        byte[] buffer = new byte[1024];
        int length;
        while ((length = is.read(buffer)) > 0) {
            os.write(buffer, 0, length);
        }
        os.close();
        is.close();
    }

    @Override
    public Object callback(@NonNull Frame frame, @Nullable Map<String, Object> arguments) {
        Log.i(TAG, "Java callback called");
        
        try {
            if (nativeModule != null) {
                // Get frame dimensions and image data (handle potential FrameInvalidError)
                int width, height;
                android.media.Image image;
                try {
                    width = frame.getWidth();
                    height = frame.getHeight();
                    image = frame.getImage();
                } catch (FrameInvalidError e) {
                    Log.e(TAG, "Frame is invalid", e);
                    return createErrorResult();
                }
                
                Log.i(TAG, "Processing frame: " + width + "x" + height);
                Log.i(TAG, "Calling nativeProcessFrameWithData with real Frame object");
                
                // Handle debug images argument
                if (arguments != null && arguments.containsKey("debugImages")) {
                    boolean debugImages = (Boolean) arguments.get("debugImages");
                    nativeModule.setDebugImages(debugImages);
                    Log.i(TAG, "Debug images " + (debugImages ? "enabled" : "disabled"));
                }
                
                // Handle useTflite argument for A/B testing
                boolean useTflite = false;
                if (arguments != null && arguments.containsKey("useTflite")) {
                    useTflite = (Boolean) arguments.get("useTflite");
                    Log.i(TAG, "Model backend: " + (useTflite ? "TFLite" : "ONNX"));
                }
                
                // Handle modelSize argument
                int modelSize = 640; // Default
                if (arguments != null && arguments.containsKey("modelSize")) {
                    modelSize = ((Number) arguments.get("modelSize")).intValue();
                    Log.i(TAG, "Model size: " + modelSize + "x" + modelSize);
                }
                
                // Handle enabledTypes argument and convert to bitmask
                int enabledTypesMask = 0x1F; // Default: all 5 classes enabled (0x01 | 0x02 | 0x04 | 0x08 | 0x10)
                if (arguments != null && arguments.containsKey("enabledTypes")) {
                    @SuppressWarnings("unchecked")
                    List<String> enabledTypes = (List<String>) arguments.get("enabledTypes");
                    enabledTypesMask = convertEnabledTypesToBitmask(enabledTypes);
                    Log.i(TAG, "Enabled types mask: 0x" + Integer.toHexString(enabledTypesMask).toUpperCase());
                }
                
                // Extract actual frame data from VisionCamera Image
                android.media.Image.Plane[] planes = image.getPlanes();
                
                // Get Y, U, V planes from YUV_420_888 format
                android.media.Image.Plane yPlane = planes[0];
                android.media.Image.Plane uPlane = planes[1];
                android.media.Image.Plane vPlane = planes[2];
                
                // Calculate total buffer size needed
                int ySize = yPlane.getBuffer().remaining();
                int uSize = uPlane.getBuffer().remaining();  
                int vSize = vPlane.getBuffer().remaining();
                int totalSize = ySize + uSize + vSize;
                
                // Create combined YUV byte array
                byte[] frameData = new byte[totalSize];
                yPlane.getBuffer().get(frameData, 0, ySize);
                uPlane.getBuffer().get(frameData, ySize, uSize);
                vPlane.getBuffer().get(frameData, ySize + uSize, vSize);
                
                Log.i(TAG, "Extracted frame data: " + frameData.length + " bytes (Y:" + ySize + " U:" + uSize + " V:" + vSize + ")");
                
                // Set model size before processing
                nativeModule.setModelSize(modelSize);
                
                // Call native processing with real frame data
                String jsonResult = nativeModule.nativeProcessFrameWithData(width, height, frameData, enabledTypesMask, useTflite);
                Log.i(TAG, "Native result: " + jsonResult);
                
                // Parse JSON result and convert to expected format
                JSONObject jsonObj = new JSONObject(jsonResult);
                
                if (jsonObj.has("error")) {
                    Log.e(TAG, "Native processing error: " + jsonObj.getString("error"));
                    return createErrorResult();
                }
                
                // Convert JSON to Java Map format expected by the UI
                Map<String, Object> result = new HashMap<>();
                
                // Process detections array (Stage 1 - for bounding boxes)
                List<Map<String, Object>> detections = new ArrayList<>();
                if (jsonObj.has("detections")) {
                    JSONArray detectionsArray = jsonObj.getJSONArray("detections");
                    
                    for (int i = 0; i < detectionsArray.length(); i++) {
                        JSONObject detection = detectionsArray.getJSONObject(i);
                        
                        Map<String, Object> detectionMap = new HashMap<>();
                        detectionMap.put("type", detection.getString("type"));
                        detectionMap.put("confidence", detection.getDouble("confidence"));
                        detectionMap.put("model", detection.getString("model"));
                        
                        // Flat structure for ModelComparisonApp compatibility
                        detectionMap.put("x", detection.getInt("x"));
                        detectionMap.put("y", detection.getInt("y"));
                        detectionMap.put("width", detection.getInt("width"));
                        detectionMap.put("height", detection.getInt("height"));
                        
                        detections.add(detectionMap);
                    }
                }
                
                // Process ocr_results array (Stage 2 - for extracted text)
                List<Map<String, Object>> ocrResults = new ArrayList<>();
                if (jsonObj.has("ocr_results")) {
                    JSONArray ocrArray = jsonObj.getJSONArray("ocr_results");
                    
                    for (int i = 0; i < ocrArray.length(); i++) {
                        JSONObject ocr = ocrArray.getJSONObject(i);
                        
                        Map<String, Object> ocrMap = new HashMap<>();
                        ocrMap.put("type", ocr.getString("type"));
                        ocrMap.put("value", ocr.getString("value"));
                        ocrMap.put("confidence", ocr.getDouble("confidence"));
                        ocrMap.put("model", ocr.getString("model"));
                        
                        ocrResults.add(ocrMap);
                    }
                }
                
                result.put("detections", detections);
                result.put("ocr_results", ocrResults);
                Log.i(TAG, "Returning " + detections.size() + " detections and " + ocrResults.size() + " OCR results from native processing");
                
                return result;
                
            } else {
                Log.w(TAG, "Native module not available, returning error");
                return createErrorResult();
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return createErrorResult();
        }
    }
    
    private Map<String, Object> createErrorResult() {
        Map<String, Object> result = new HashMap<>();
        List<Map<String, Object>> detections = new ArrayList<>();
        result.put("detections", detections);
        return result;
    }
    
    /**
     * Convert enabledTypes string array to bitmask for efficient native processing
     * Matches the constants defined in CodeDetectionConstants.h
     */
    private int convertEnabledTypesToBitmask(List<String> enabledTypes) {
        if (enabledTypes == null || enabledTypes.isEmpty()) {
            return 0x1F; // All 5 classes enabled by default
        }
        
        int mask = 0;
        for (String type : enabledTypes) {
            switch (type) {
                case "code_container_h":
                    mask |= 0x01; // CONTAINER_H = bit 0
                    break;
                case "code_container_v":
                    mask |= 0x02; // CONTAINER_V = bit 1
                    break;
                case "code_license_plate":
                    mask |= 0x04; // LICENSE_PLATE = bit 2
                    break;
                case "code_qr_barcode":
                    mask |= 0x08; // QR_BARCODE = bit 3
                    break;
                case "code_seal":
                    mask |= 0x10; // SEAL = bit 4
                    break;
                default:
                    Log.w(TAG, "Unknown code detection type: " + type);
                    break;
            }
        }
        
        return mask;
    }
}