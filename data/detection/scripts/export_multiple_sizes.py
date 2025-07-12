#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX at multiple sizes (320, 416, 640)
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def export_model_sizes():
    """Export trained model to different ONNX sizes"""
    
    # Load the best model from training
    # First try to find the .pt file
    possible_paths = [
        Path("../detection/models/unified-detection-v7.pt")
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            print(f"Found model at: {model_path}")
            break
    
    if model_path is None:
        print("Error: Could not find trained model (.pt file)")
        print("Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease ensure you have a trained YOLOv8 model.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path("../detections/models/onnx")
    output_dir.mkdir(exist_ok=True)
    
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Export at different sizes
    sizes = [320, 416, 640]
    
    for size in sizes:
        print(f"\nExporting model at {size}x{size}...")
        
        try:
            # Export to ONNX with specific image size
            onnx_path = model.export(
                format='onnx',
                imgsz=size,
                simplify=True,
                opset=12,  # Compatible with ONNX Runtime Mobile
                batch=1,
                dynamic=False  # Fixed size for mobile optimization
            )
            
            # Rename to include size in filename
            output_path = output_dir / f"unified-detection-v7-{size}.onnx"
            
            # Handle the case where export returns a Path object or string
            if isinstance(onnx_path, Path):
                onnx_path.rename(output_path)
            else:
                os.rename(str(onnx_path), str(output_path))
            
            print(f"✅ Exported: {output_path}")
            print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        except Exception as e:
            print(f"❌ Error exporting at {size}x{size}: {e}")
            continue
    
    print("\n🎉 All models exported successfully!")
    print("\nNext steps:")
    print("1. Copy models to Android assets:")
    print("   cp ../dection/models/onnx/unified-detection-v7-*.onnx ../../android/src/main/assets/")
    print("2. Update C++ code to load the appropriate model based on size parameter")

if __name__ == "__main__":
    export_model_sizes()