package com.universal;

import com.facebook.jni.HybridData;
import com.facebook.soloader.SoLoader;
import android.util.Log;
import android.content.Context;

public class UniversalNativeModule {
    private static final String TAG = "UniversalNative";
    
    @SuppressWarnings("unused")
    private final HybridData mHybridData;
    
    static {
        try {
            SoLoader.loadLibrary("universal");
            Log.i(TAG, "Native library loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load native library", e);
        }
    }
    
    public UniversalNativeModule() {
        mHybridData = initHybrid();
    }
    
    private native HybridData initHybrid();
    
    public native String nativeProcessFrame(int width, int height);
    
    public native String nativeProcessFrameWithData(int width, int height, byte[] frameData, int enabledTypesMask);
    
    public native void setDebugImages(boolean enabled);
    
}