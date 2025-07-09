#!/bin/bash

# Script to pull OCR debug images from Android device

echo "📸 Pulling OCR debug images from device..."

# Create debug_images directory if it doesn't exist
mkdir -p debug_images

# Clean up old images
rm -rf debug_images/*

# Pull the entire onnx_debug directory
adb pull /sdcard/Download/onnx_debug/ debug_images/ 2>/dev/null

# If nested directory was created, move files up
if [ -d "debug_images/onnx_debug" ]; then
    mv debug_images/onnx_debug/* debug_images/ 2>/dev/null
    rmdir debug_images/onnx_debug 2>/dev/null
fi

echo "✅ Debug images saved to debug_images/"
echo ""

# List the pulled files
echo "Files pulled:"
ls -la debug_images/*.jpg 2>/dev/null | awk '{print "  " $9}'

echo ""
echo "OCR Pipeline stages:"
echo "  *_0_ocr_yuv_crop.jpg      - Raw YUV crop from detection bbox"
echo "  *_1_ocr_rgb_converted.jpg - After YUV to RGB conversion"
echo "  *_2_ocr_rotated.jpg       - After 90° CW rotation"
echo "  *_3_ocr_scaled.jpg        - After resize to 640 longest dimension"
echo "  *_4_ocr_final_padded.jpg  - Final 640x640 padded image sent to OCR"
echo ""
echo "Detection Pipeline stages:"
echo "  *_0_original_yuv.jpg      - Original YUV frame"
echo "  *_0b_resized_yuv.jpg      - Resized YUV frame"
echo "  *_1_rgb_converted.jpg     - After YUV to RGB conversion"
echo "  *_2_rotated.jpg           - After 90° CW rotation"
echo "  *_3_padded.jpg            - After padding to square"