# solve image to ocr pipeline issue

Log shows:
```
07-10 11:15:50.525 17965 18041 I UniversalScanner: 🔄 Starting preprocessing phase...
07-10 11:15:50.525 17965 18041 I UniversalScanner: Processing REAL camera frame data: 1280x720, 1843198 bytes
07-10 11:15:50.568 17965 18042 I UniversalScanner: Native module initialized successfully
07-10 11:15:50.605 17965 18042 I UniversalScanner: Native module initialized successfully
07-10 11:15:50.630 17965 18041 I UniversalScanner: Resizing YUV: 1280x720 -> 320x180
07-10 11:15:50.655 17965 18041 I UniversalScanner: Converting YUV to RGB (320x180)
07-10 11:15:50.680 17965 18041 I UniversalScanner: Applying 90° CW rotation (320x180)
07-10 11:15:50.694 17965 18041 I UniversalScanner: ✅ Preprocessing complete: 180x320
07-10 11:15:50.694 17965 18041 I UniversalScanner: 🧮 Creating tensor from RGB data...
07-10 11:15:50.694 17965 18041 I UniversalScanner: Creating tensor from RGB data (180x320)
07-10 11:15:50.729 17965 18041 I UniversalScanner: Tensor created: 320x320, first pixels: 0.263 0.278 0.251
07-10 11:15:50.729 17965 18041 I UniversalScanner: ✅ Tensor created: 307200 elements
07-10 11:15:50.729 17965 18041 I UniversalScanner: 🧠 Starting ONNX inference phase...
07-10 11:15:50.729 17965 18041 I UniversalScanner: 🚀 Running ONNX inference with cpu provider
07-10 11:15:50.857 17965 18041 I UniversalScanner: ✅ ONNX inference completed, output shape: [1, 9, 2100]
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🔬 First 10 output values: 44.641922 42.015030 40.271980 39.325897 37.413101 38.907455 40.120056 42.321911 82.428001 84.002319
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🔍 Processing 2100 anchors for best detection with enabled types mask: 0x03
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🔬 First anchor sample:
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 0: 44.641922
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 1: 2.955750
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 2: 91.001175
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 3: 5.632656
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 4: 0.000023
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 5: 0.000021
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 6: 0.000586
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 7: 0.000068
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 8: 0.000177
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🎯 High confidence anchor 0: class=0, conf=0.500, bbox=(44.642,2.956,91.001,5.633)
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🎯 High confidence anchor 10: class=0, conf=0.500, bbox=(84.408,2.764,179.057,5.762)
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🎯 High confidence anchor 488: class=0, conf=0.500, bbox=(65.765,106.130,12.658,20.300)
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🎯 High confidence anchor 528: class=1, conf=0.500, bbox=(65.991,108.799,12.120,26.090)
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🎯 High confidence anchor 562: class=0, conf=0.500, bbox=(36.902,118.628,74.115,23.654)
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🎯 High confidence anchor 568: class=1, conf=0.500, bbox=(66.100,113.596,12.269,39.053)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 602: class=0, conf=0.500, bbox=(35.759,121.204,71.734,20.843)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 648: class=1, conf=0.500, bbox=(66.168,121.073,11.342,55.046)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 688: class=1, conf=0.501, bbox=(66.192,129.212,11.256,69.730)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 728: class=1, conf=0.503, bbox=(65.993,144.914,11.373,105.143)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 968: class=1, conf=0.505, bbox=(66.787,186.217,17.232,177.854)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 1804: class=1, conf=0.518, bbox=(66.325,160.007,16.703,132.903)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 1824: class=1, conf=0.521, bbox=(65.989,159.397,18.370,128.186)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 📊 Detection stats: 2100 valid (>0.1), 1931 high-conf (>0.5), 1 passed threshold (>0.5)
07-10 11:15:50.858 17965 18041 I UniversalScanner: ✅ Best detection: class=1 (code_container_v), conf=0.521, coords=(0.206,0.498) size=0.057x0.401
07-10 11:15:50.858 17965 18041 I UniversalScanner: ✅ ONNX inference complete
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🚀 Frame processed using cpu (332.65 ms total)
07-10 11:15:50.858 17965 18041 I UniversalScanner: ⚠️ SLOW FRAME: 332.66 ms (target: <33ms for 30 FPS) with cpu
07-10 11:15:50.858 17965 18041 I UniversalNative: ✅ Stage 1 complete: code_container_v conf=0.521
07-10 11:15:50.858 17965 18041 I UniversalNative: 🎯 Stage 1 emit: Immediate detection feedback
07-10 11:15:50.858 17965 18041 I UniversalNative: 🎯 Stage 1 response: {"detections":[{"type":"code_container_v","confidence":0.520818,"x":65,"y":159,"width":18,"height":128,"model":"unified-detection-v7-320.onnx"}],"ocr_results":[],"ocr_status":"not_attempted"}
07-10 11:15:50.858 17965 18041 I UniversalNative: 📞 Stage 2: Running OCR pipeline (confidence 0.521 > 0.51)...
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐 Coordinate conversion per spec:
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   Input: norm(0.206,0.498,0.057,0.401)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   Rotated: (0.498,0.206,0.401,0.057)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   FrameSize: 1280 (max of 1280x720)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   Before padding: (381,227,512,73)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   After padding: (371,187,532,153) in frame space
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 🎯 Final crop region: (372,188,532,152) for code_container_v
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: ✂️ Extracted YUV crop: 532x152 from frame 1280x720
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📸 Debug images enabled: YES
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📸 Saving OCR debug image: container_v_0_ocr_yuv_crop.jpg
07-10 11:15:50.871 17965 18041 D OnnxProcessorV2: 🔧 Converting YUV to RGB at full resolution 532x152
07-10 11:15:50.880 17965 18041 D OnnxProcessorV2: 📸 Saving OCR debug image: container_v_1_ocr_rgb_converted.jpg
07-10 11:15:50.891 17965 18041 D OnnxProcessorV2: 🔧 Rotating 90° CW for OCR (532x152 → 152x532)
07-10 11:15:50.893 17965 18041 D OnnxProcessorV2: 📸 Saving OCR debug image: container_v_2_ocr_rotated.jpg
07-10 11:15:50.902 17965 18041 D OnnxProcessorV2: 🔧 Resizing 152x532 to 182x640 (longest dimension to 640)
07-10 11:15:50.912 17965 18041 D OnnxProcessorV2: 📸 Saving OCR debug image: container_v_3_ocr_scaled.jpg
07-10 11:15:50.925 17965 18041 D OnnxProcessorV2: 🔧 Padding to 640x640 with white
07-10 11:15:50.927 17965 18041 D OnnxProcessorV2: 📸 Saving OCR debug image: container_v_4_ocr_final_padded.jpg
07-10 11:15:50.954 17965 18041 D OnnxProcessorV2: ✅ OCR preprocessing complete: 640x640 padded image
07-10 11:15:50.954 17965 18041 D YoloOCREngine: 🔤 OCR recognize: image 640x640 (1228800 bytes) for class code_container_v
07-10 11:15:50.954 17965 18041 D YoloOCREngine: 🔤 Converting image to tensor...
07-10 11:15:50.977 17965 18041 D YoloOCREngine: 🔤 Tensor created: 1228800 elements
07-10 11:15:50.977 17965 18041 D YoloOCREngine: 🔤 Running ONNX inference...
07-10 11:15:51.293 17965 18041 D YoloOCREngine: 🔤 ONNX inference completed
07-10 11:15:51.293 17965 18041 D YoloOCREngine: 🔤 Parsing YOLO output...
07-10 11:15:51.296 17965 18041 D YoloOCREngine: Parsed 0 character detections from YOLO output
07-10 11:15:51.296 17965 18041 D YoloOCREngine: 🔤 Found 0 character boxes
07-10 11:15:51.296 17965 18041 D YoloOCREngine: NMS: 0 boxes -> 0 boxes
07-10 11:15:51.297 17965 18041 D YoloOCREngine: Sorting characters vertically (top to bottom) for code_container_v
07-10 11:15:51.297 17965 18041 D YoloOCREngine: Assembled text: 
07-10 11:15:51.297 17965 18041 D OnnxProcessorV2: OCR result: '' (conf: 0.00)
07-10 11:15:51.297 17965 18041 I UniversalNative: 📞 Stage 2 complete: 1 OCR results
07-10 11:15:51.297 17965 18041 I UniversalNative: 🎯 Stage 2 emit: Enhanced response with OCR results
07-10 11:15:51.297 17965 18041 I UniversalNative: 🎯 Progressive enhancement architecture enabled
07-10 11:15:51.297 17965 18041 I UniversalNative: 🎯 Returning final response: {"detections":[{"type":"code_container_v","confidence":0.520818,"x":65,"y":159,"width":18,"height":128,"model":"unified-detection-v7-320.onnx"}],"ocr_results":[{"type":"code_container_v","value":"","confidence":0.000000,"model":"yolo-ocr-v7-640"}],"ocr_status":"completed"}
07-10 11:15:51.297 17965 18041 I UniversalScanner: Native result: {"detections":[{"type":"code_container_v","confidence":0.520818,"x":65,"y":159,"width":18,"height":128,"model":"unified-detection-v7-320.onnx"}],"ocr_results":[{"type":"code_container_v","value":"","confidence":0.000000,"model":"yolo-ocr-v7-640"}],"ocr_status":"completed"}
07-10 11:15:51.297 17965 18041 I UniversalScanner: Returning 1 detections and 1 OCR results from native processing
07-10 11:15:51.300 17965 18042 I ReactNativeJS: 'OCR Results:', [ { confidence: 0,
07-10 11:15:51.300 17965 18042 I ReactNativeJS: Merged OCR value "" into detection of type code_container_v
```

## Image flow:

### preprosessing
![step 0: input](debug_images/ocr_only_R3CX70KQ5RZ/111536_0_original_yuv.jpg)

07-10 11:15:50.630 17965 18041 I UniversalScanner: Resizing YUV: 1280x720 -> 320x180

![step 1: resized](debug_images/ocr_only_R3CX70KQ5RZ/111536_0b_resized_yuv.jpg)

07-10 11:15:50.655 17965 18041 I UniversalScanner: Converting YUV to RGB (320x180)

![step 2: rgb converted](debug_images/ocr_only_R3CX70KQ5RZ/111536_1_rgb_converted.jpg)

07-10 11:15:50.680 17965 18041 I UniversalScanner: Applying 90° CW rotation (320x180)

![step 3: rotated](debug_images/ocr_only_R3CX70KQ5RZ/111536_2_rotated.jpg)

07-10 11:15:50.694 17965 18041 I UniversalScanner: 🧮 Creating tensor from RGB data...
07-10 11:15:50.694 17965 18041 I UniversalScanner: Creating tensor from RGB data (180x320)
07-10 11:15:50.729 17965 18041 I UniversalScanner: Tensor created: 320x320, first pixels: 0.263 0.278 0.251

![step 4: padded](debug_images/ocr_only_R3CX70KQ5RZ/111536_3_padded.jpg)

### Inference

07-10 11:15:50.729 17965 18041 I UniversalScanner: ✅ Tensor created: 307200 elements
07-10 11:15:50.729 17965 18041 I UniversalScanner: 🧠 Starting ONNX inference phase...
07-10 11:15:50.729 17965 18041 I UniversalScanner: 🚀 Running ONNX inference with cpu provider
07-10 11:15:50.857 17965 18041 I UniversalScanner: ✅ ONNX inference completed, output shape: [1, 9, 2100]

### Post processing

```
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🔍 Processing 2100 anchors for best detection with enabled types mask: 0x03
07-10 11:15:50.857 17965 18041 I UniversalScanner: 🔬 First anchor sample:
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 0: 44.641922
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 1: 2.955750
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 2: 91.001175
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 3: 5.632656
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 4: 0.000023
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 5: 0.000021
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 6: 0.000586
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 7: 0.000068
07-10 11:15:50.857 17965 18041 I UniversalScanner:    Feature 8: 0.000177
07-10 11:15:50.858 17965 18041 I UniversalScanner: 🎯 High confidence anchor 688: class=1, conf=0.501, bbox=(66,129,11,69)
07-10 11:15:50 17965 18041 I UniversalScanner: 🎯 High confidence anchor 728: class=1, conf=0.503, bbox=(65,144,11,105)
07-10 11:15:50 17965 18041 I UniversalScanner: 🎯 High confidence anchor 968: class=1, conf=0.505, bbox=(66,186,17,177)
07-10 11:15:50 17965 18041 I UniversalScanner: 🎯 High confidence anchor 1804: class=1, conf=0.518, bbox=(66,160,16,132)
07-10 11:15:50 17965 18041 I UniversalScanner: 🎯 High confidence anchor 1824: class=1, conf=.523, bbox=(65,159,18,128)
07-10 11:15:50.858 17965 18041 I UniversalScanner: 📊 Detection stats: 2100 valid (>0.1), 1931 high-conf (>0.5), 1 passed threshold (>0.5)
07-10 11:15:50.858 17965 18041 I UniversalScanner: ✅ Best detection: class=1 (code_container_v), conf=0.521, coords=(0.206,0.498) size=0.057x0.401
07-10 11:15:50.858 17965 18041 I UniversalScanner: ✅ ONNX inference complete
```

This is correct, per check calculation:
Converting bbox (65.989, 159.397,
  18.370, 128.186) from 320x320 space
  to relative percentages:

  Relative bbox: (0.2062, 0.4981,
  0.0574, 0.4006)

  Calculation:
  - x% = 65 / 320 = 0.206
  - y% = 159 / 320 = 0.498
  - w% = 18 / 320 = 0.057
  - h% = 128 / 320 = 0.401

Corresponds with this area:
![bbox](image.png)
It's not entirely correct, but for all intents and purposes, lets go with it!

# OCR pipeline
Moving to 

![Step 1:crop ](debug_images/ocr_only_R3CX70KQ5RZ/111536_container_v_0_ocr_yuv_crop.jpg)

here we immediately go wrong! I'd expect something looking like our last image; but this ain't that!

Lets check the log:
```
07-10 11:15:50.858 17965 18041 I UniversalNative: 🎯 Stage 1 response: {"detections":[{"type":"code_container_v","confidence":0.520818,"x":65,"y":159,"width":18,"height":128,"model":"unified-detection-v7-320.onnx"}],"ocr_results":[],"ocr_status":"not_attempted"}
07-10 11:15:50.858 17965 18041 I UniversalNative: 📞 Stage 2: Running OCR pipeline (confidence 0.521 > 0.51)...
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐 Coordinate conversion per spec:
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   Input: norm(0.206,0.498,0.057,0.401)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   Rotated: (0.498,0.206,0.401,0.057)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   FrameSize: 1280 (max of 1280x720)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   Before padding: (381,227,512,73)
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 📐   After padding: (371,187,532,153) in frame space
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: 🎯 Final crop region: (372,188,532,152) for code_container_v
07-10 11:15:50.858 17965 18041 D OnnxProcessorV2: ✂️ Extracted YUV crop: 532x152 from frame 1280x720
```

