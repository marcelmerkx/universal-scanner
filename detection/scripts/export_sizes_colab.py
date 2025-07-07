"""
Colab script to export YOLOv8 model at multiple sizes
Upload your unified-detection-v7.pt file to Colab first
"""

# Install required packages
!pip install ultralytics onnx onnxsim

from ultralytics import YOLO
import os

# Load the model (upload unified-detection-v7.pt to Colab first)
model_path = 'unified-detection-v7.pt'

if not os.path.exists(model_path):
    print(f"Please upload {model_path} to Colab first!")
    print("You can drag and drop the file into the file browser on the left")
else:
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Export at different sizes
    sizes = [320, 416, 640]
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Exporting model at {size}x{size}...")
        
        # Export to ONNX
        onnx_path = model.export(
            format='onnx',
            imgsz=size,
            simplify=True,
            opset=12,
            batch=1,
            dynamic=False
        )
        
        # Rename to include size
        output_name = f'unified-detection-v7-{size}.onnx'
        os.rename(onnx_path, output_name)
        
        print(f"✅ Exported: {output_name}")
        print(f"   Size: {os.path.getsize(output_name) / 1024 / 1024:.2f} MB")
    
    print("\n🎉 All models exported!")
    print("\nDownload the files:")
    print("- unified-detection-v7-320.onnx")
    print("- unified-detection-v7-416.onnx") 
    print("- unified-detection-v7-640.onnx")
    print("\nThen copy them to:")
    print("- android/src/main/assets/")
    print("- example/assets/")