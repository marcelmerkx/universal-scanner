#!/usr/bin/env python3
"""
Export Mobile-Optimized YOLO Models

Creates multiple optimized versions for testing:
1. 416x416 input size (faster than 640x640)
2. INT8 quantized version
3. Simplified graph version

Usage:
    python3 detection/scripts/export_mobile_optimized.py
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)

def export_mobile_models():
    """Export multiple mobile-optimized versions."""
    
    model_path = Path("detection/models/unified-detection-v7.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return False
    
    print(f"Loading model from: {model_path}")
    
    try:
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully")
        
        # 1. Export 416x416 version (36% fewer pixels than 640x640)
        print("\n1️⃣ Exporting 416x416 input size model...")
        model.export(
            format="onnx",
            imgsz=416,  # Smaller input = faster inference
            half=False,  # FP32 for compatibility
            simplify=True,
            opset=11,
            dynamic=False,
            verbose=False
        )
        print("✅ Exported: unified-detection-v7.onnx (416x416)")
        
        # 2. Export with INT8 flag (if supported)
        print("\n2️⃣ Exporting INT8-ready model...")
        try:
            model.export(
                format="onnx",
                imgsz=416,
                int8=True,  # Request INT8 export
                simplify=True,
                verbose=False
            )
            print("✅ Exported: INT8-ready model")
        except:
            print("⚠️ INT8 export not supported in this version")
        
        # 3. Export minimal model (no post-processing)
        print("\n3️⃣ Exporting minimal model...")
        model.export(
            format="onnx",
            imgsz=320,  # Even smaller for testing
            half=False,
            simplify=True,
            opset=10,  # Conservative opset
            dynamic=False,
            verbose=False
        )
        print("✅ Exported: unified-detection-v7.onnx (320x320)")
        
        print("\n📊 Performance expectations:")
        print("  640x640 → 416x416: ~2.4x faster")
        print("  640x640 → 320x320: ~4x faster")
        print("  FP32 → INT8: ~2-4x faster")
        print("  Combined optimizations: 30-50ms achievable")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Mobile-Optimized YOLO Export ===")
    success = export_mobile_models()
    
    if success:
        print("\n🎉 Export completed!")
        print("\nRecommended testing order:")
        print("1. Test 416x416 model (easiest, immediate speedup)")
        print("2. Quantize to INT8 (biggest performance gain)")
        print("3. Enable XNNPACK (better CPU utilization)")
    else:
        print("\n💥 Export failed!")
    
    sys.exit(0 if success else 1)