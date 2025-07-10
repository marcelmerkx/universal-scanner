# Fix pipeline for OCR

## Background
at this stage, we have a LANDSCAPE camera frame (1280x720) and on that frame we are looking for the bounding boxes of the highest confidence class of objects per a YOLO detection.
It's not going too smooth, hence this file, we're going to systematically go through each step, tracking the logic and making sure the pipeline functions optimally. 

We're almost there.  We will want to take 2 more steps: 
* we have a new weights model, which was trained on 320x320 grayscale images. So in our preprocessing pipeline we need to not  just go from YUV to RGB, we need to make it grayscale (but for ONNX YOLO use, to I guess still using 3 channels in RGB, just gray). 
* improve current flow, we make a lot of misses still

## DectionPreprocessing
Each chapter be a function with a single job to do. 
Handled in file cpp/OnnxProcessor.cpp
To be called in sequence

### STEP 1: resizeYuvForPerformance
**Explanation**: make image much smaller in order to handle next steps more efficiently too

**Input**: YuvFrame, frameWidth, frameHeight, targetWidth, targetHeight

**Output**: resizedYuvFrame

**example**: 
we receive a 1280x720 frame, we convert it to a 320x180 frame


### STEP 2: convertYuvToGrayscaleRgb
**Explanation**: converts from YUV format to grayscale RGB (3-channel format where R=G=B=Y)

**input**: resizedYuvFrame 

**output**: resizedGrayscaleRgbFrame (3-channel RGB format with identical values in each channel)

**Example**: 
- Extract Y (luminance) channel from YUV data
- Replicate Y value across all three RGB channels (R=G=B=Y)
- This is more efficient than YUV→RGB→Grayscale conversion
- Maintains compatibility with ONNX models expecting 3-channel input 

### STEP 3: rotateFrameCW
**Explanation**: rotates the image 90 degrees clockwise

**input**: resizedGrayscaleRgbFrame 

**output**: rotatedResizedGrayscaleRgbFrame

**Example**: 

### STEP 4: rightPadFrame
**Explanation**: Image must be square; we will right-pad the image with white padding. 

**input**: rotatedResizedGrayscaleRgbFrame 

**output**: paddedRotatedResizedGrayscaleRgbFrame

## Detection
**Explanation**: provide the pre-processed image to ONNX runtime, for inference

**input**: paddedRotatedResizedGrayscaleRgbFrame 

**output**: ONNX detections. 


## DetectionPostProcessing
**Explanation**: The ONNX output is complex (!), and contains many potential areas of interest, with a confidence score. Goal for this step is to find that area containing the actual class that was detected. 

**input**: 2100 ONNX detections (for a 320x320 image)

**output**: one ONNX detection in the struct DetectionResult format

