#!/usr/bin/env python3
"""
Simple script to export YOLOv8 model to ONNX at multiple sizes
Run from the detection/scripts directory
"""
import os
import sys

# Check if we're in the right directory
if not os.path.exists('../models/unified-detection-v7.pt'):
    print("Error: Please run this script from the detection/scripts directory")
    print("Current directory:", os.getcwd())
    sys.exit(1)

# Check for required packages
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found")
    print("Please install with: pip install ultralytics")
    sys.exit(1)

print("Loading model from ../models/unified-detection-v7.pt")
model = YOLO('../models/unified-detection-v7.pt')

# Export at different sizes
sizes = [320, 416, 640]

for size in sizes:
    print(f"\n{'='*50}")
    print(f"Exporting model at {size}x{size}...")
    
    try:
        # Export to ONNX
        # The export method returns the path where it saved the file
        result = model.export(
            format='onnx',
            imgsz=size,
            simplify=True,
            opset=12,
            batch=1,
            dynamic=False
        )
        
        print(f"Export completed. File saved at: {result}")
        
        # The exported file is typically saved next to the .pt file
        # Let's move it to our desired location
        import shutil
        source_path = str(result)
        dest_path = f'../models/unified-detection-v7-{size}.onnx'
        
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print(f"✅ Moved to: {dest_path}")
            
            # Check file size
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
        else:
            print(f"⚠️  Warning: Could not find exported file at {source_path}")
            
    except Exception as e:
        print(f"❌ Error exporting {size}x{size}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("Export process completed!")
print("\nTo use these models:")
print("1. Copy to Android assets:")
print("   cp ../models/unified-detection-v7-*.onnx ../../android/src/main/assets/")
print("2. Copy to example assets:")
print("   cp ../models/unified-detection-v7-*.onnx ../../example/assets/")