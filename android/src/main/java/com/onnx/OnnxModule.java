package com.onnx;

import com.facebook.react.bridge.JavaScriptContextHolder;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;
import com.facebook.react.turbomodule.core.interfaces.CallInvokerHolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

@ReactModule(name = OnnxModule.NAME)
public class OnnxModule extends ReactContextBaseJavaModule {
  public static final String NAME = "RNOnnx";

  static {
    System.loadLibrary("onnx");
  }

  public OnnxModule(ReactApplicationContext reactContext) {
    super(reactContext);
  }

  @Override
  public String getName() {
    return NAME;
  }

  @ReactMethod(isBlockingSynchronousMethod = true)
  public boolean install() {
    try {
      JavaScriptContextHolder jsContext = getReactApplicationContext().getJavaScriptContextHolder();
      CallInvokerHolder callInvokerHolder = getReactApplicationContext().getCatalystInstance().getJSCallInvokerHolder();
      
      return nativeInstall(jsContext.get(), (CallInvokerHolderImpl) callInvokerHolder);
    } catch (Exception exception) {
      return false;
    }
  }

  public static byte[] fetchByteDataFromUrl(String url) throws IOException {
    InputStream inputStream = new URL(url).openStream();
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    int nRead;
    byte[] data = new byte[1024];
    while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
      buffer.write(data, 0, nRead);
    }
    buffer.flush();
    return buffer.toByteArray();
  }

  public static native boolean nativeInstall(long jsiPtr, CallInvokerHolderImpl jsCallInvokerHolder);
}