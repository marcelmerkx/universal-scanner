package com.tflite;

import androidx.annotation.NonNull;

import com.facebook.react.ReactPackage;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.ViewManager;

// VisionCamera Frame Processor Plugin registration
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry;
import com.universal.UniversalScannerFrameProcessorPlugin;

// OnnxModule is now in the same package

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TflitePackage implements ReactPackage {
  
  // Static initializer to register VisionCamera Frame Processor Plugin
  static {
    FrameProcessorPluginRegistry.addFrameProcessorPlugin("universalScanner", (proxy, options) -> {
      return new UniversalScannerFrameProcessorPlugin(proxy);
    });
  }

  @NonNull
  @Override
  public List<NativeModule> createNativeModules(@NonNull ReactApplicationContext reactContext) {
    // Set the application context for the frame processor plugin
    UniversalScannerFrameProcessorPlugin.setApplicationContext(reactContext);
    
    List<NativeModule> modules = new ArrayList<>();
    modules.add(new TfliteModule(reactContext));
    modules.add(new OnnxModule(reactContext)); // Also register ONNX module
    return modules;
  }

  @NonNull
  @Override
  public List<ViewManager> createViewManagers(@NonNull ReactApplicationContext reactContext) {
    return Collections.emptyList();
  }
}
