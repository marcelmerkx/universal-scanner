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
            // Extract unified-detection-v7.onnx from assets to internal storage
            java.io.InputStream is = appContext.getAssets().open("unified-detection-v7.onnx");
            java.io.File outputFile = new java.io.File(appContext.getFilesDir(), "unified-detection-v7.onnx");
            
            java.io.OutputStream os = new java.io.FileOutputStream(outputFile);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
            os.close();
            is.close();
            
            modelExtracted = true;
            Log.i(TAG, "Model extracted to: " + outputFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Failed to extract model from assets", e);
        }
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
                
                // Call native processing with real frame data
                String jsonResult = nativeModule.nativeProcessFrameWithData(width, height, frameData);
                Log.i(TAG, "Native result: " + jsonResult);
                
                // Parse JSON result and convert to expected format
                JSONObject jsonObj = new JSONObject(jsonResult);
                
                if (jsonObj.has("error")) {
                    Log.e(TAG, "Native processing error: " + jsonObj.getString("error"));
                    return createErrorResult();
                }
                
                // Convert JSON to Java Map format expected by the UI
                Map<String, Object> result = new HashMap<>();
                List<Map<String, Object>> detections = new ArrayList<>();
                
                if (jsonObj.has("detections")) {
                    JSONArray detectionsArray = jsonObj.getJSONArray("detections");
                    
                    for (int i = 0; i < detectionsArray.length(); i++) {
                        JSONObject detection = detectionsArray.getJSONObject(i);
                        
                        Map<String, Object> detectionMap = new HashMap<>();
                        detectionMap.put("type", detection.getString("type"));
                        detectionMap.put("confidence", detection.getDouble("confidence"));
                        detectionMap.put("width", detection.getInt("width"));
                        detectionMap.put("height", detection.getInt("height"));
                        detectionMap.put("x", detection.getInt("x"));
                        detectionMap.put("y", detection.getInt("y"));
                        detectionMap.put("model", detection.getString("model"));
                        
                        detections.add(detectionMap);
                    }
                }
                
                result.put("detections", detections);
                Log.i(TAG, "Returning " + detections.size() + " detections from native processing");
                
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
}