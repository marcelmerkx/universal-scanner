# Fix pipeline for OCR

## Background
at this stage, we have a LANDSCAPE camera frame (1280x720) and we've completed a "Detection" (AKA "Stage 1"), giving us confidence (e.g. 0.590), a class (example: code_container_v) and coordinates and size e.g. coords=(0.254,0.470) size=0.037x0.437 based on a right-padded PORTRAIT 320x320 image.

Log will show:
```
07-09 10:17:11.412  9175  9277 I UniversalScanner: 📊 Detection stats: 2100 valid (>0.1), 1888 high-conf (>0.5), 1 passed threshold (>0.55)
07-09 10:17:11.412  9175  9277 I UniversalScanner: ✅ Best detection: class=1 (code_container_v), conf=0.590, coords=(0.254,0.470) size=0.037x0.437
07-09 10:17:11.412  9175  9277 I UniversalScanner: ✅ ONNX inference complete
07-09 10:17:11.412  9175  9277 I UniversalScanner: 🚀 Frame processed using cpu (31.64 ms total)
07-09 10:17:11.412  9175  9277 D OnnxProcessorV2: 🔧 Coordinate conversion: norm(0.254,0.470,0.037,0.437) → pixel(75,80,11,139) in 180x320 space
```

## OCRPreprocessing
Each chapter be a function with a single job to do. To be called in sequence

### STEP 1: coordinateConversion
**Explanation**: convert the "detection" coordinates back to the common input Frame

**Input**: detectionResult.x, detectionResult.y, detectionResult.w, detectionResult.h, FrameWidth, FrameHeight

**Output**: x,y,w,h in the space of (FrameWidth x FrameHeight)

**Calculation** (pseudo code):
the norm results on the square inference frame should be 'mapped' to the frame. 
```
// rotate detection
rotated_x = detectionResult.y
rotated_y = detectionResult.x
rotated_w = detectionResult.h
rotated_h = detectionResult.w

// determing input frame space
FrameSize = max(FrameWidth, FrameHeight)

// return calculation
return([
    x = round(rotated_x * FrameSize),
    y = round(rotated_y * FrameSize),
    w = round(rotated_w * FrameSize),
    h = round(rotated_h * FrameSize) + 100,
])
```
**example**: this would lead to:

result = coordinateConversion(0.254,0.470,0.037,0.437,1280,720)
output:
x = 0.470 * 1280 = 602
y = 0.254 * 1280 = 325
w = 0.437 * 1280 = 589
h = 0.037 * 1280 + 100 = 147 // temporarily, we will increase the h with 100px to handle an issue in the emulator, to be removed for production!

### STEP 2: cropForOcr
**Explanation**: this takes the actual correctly spaced coordinates x,y,w,h and crops the frame to that

**input**: YUVframe , coordinateConversion.output, paddingHeight, paddingWidth

**output**: YUVframeCropped -> this is a high-resolution frame from the original

**Example**: we input the 1280 x 720 frame and we get a padded 588 x 147 YUV format back

### STEP 3: convertYuvToRgbForOcr
**Explanation**: converts from YUV to RGB format

**input**: YUVframeCropped

**output**: RGBframeCropped

**Example**: returns same file, converted to RGB. still 588 x 147

### STEP 4: rotateCropForOcr
**Explanation**: This will rotate the image, clockwise by 90 degrees

**input**: RGBframeCropped

**output**: RGBframeCroppedRotated

put in our RGB image, get it back, rotated, so we come to a 147 x 588 RGB image

#### STEP 5: resizeForOcr
**Explanation**: This will bring the image to the 640x640 space. 

**input**: RGBframeCroppedRotated, targetSize

**output**: RGBframeCroppedRotatedResized

**Example**: Finds the longest space and stretches (or shrink) to image to targetSize. I.e. in this case we will see our input at 147 x 588 and thus get a 160 x 640 image back

### STEP 6: padForOcr
**Explanation**: This will right or bottom-pad the image, to make it 640x640. Bottom padding will happen when the image is 640 wide, while right-padding happens on a 640 height image.
input: RGBframeCroppedRotatedResized

**input**: RGBframeCroppedRotatedResized, targetSize

**output**: readyForOcr

**Example**: we input our 160 x 640 image and we get it back with white right-padding at 640x640