#!/bin/bash

# Script to pull OCR debug images from all Android devices

echo "📸 Pulling OCR debug images from all devices..."

# Check for connected devices
DEVICES=$(adb devices | grep -E "device$|emulator" | cut -f1)
DEVICE_COUNT=$(echo "$DEVICES" | wc -l | tr -d ' ')

if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "❌ No devices connected!"
    exit 1
fi

# Process each device
TOTAL_IMAGES=0
for DEVICE in $DEVICES; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📱 Processing device: $DEVICE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ADB_CMD="adb -s $DEVICE"
    
    # Create device-specific directory
    OUTPUT_DIR="debug_images/ocr_only_${DEVICE}"
    mkdir -p "$OUTPUT_DIR"
    
    # Clean up old OCR images for this device
    rm -rf "$OUTPUT_DIR"/*
    
    # Check if onnx_debug directory exists
    if ! $ADB_CMD shell "test -d /sdcard/Download/onnx_debug && echo exists" | grep -q "exists"; then
        echo "⚠️  No debug directory found on $DEVICE"
        continue
    fi
    
    # List OCR files on device
    echo "🔍 Finding OCR images..."
    OCR_FILES=$($ADB_CMD shell "ls /sdcard/Download/onnx_debug/*_ocr_*.jpg 2>/dev/null" | tr -d '\r')
    # OCR_FILES=$($ADB_CMD shell "ls /sdcard/Download/onnx_debug/*.jpg 2>/dev/null" | tr -d '\r')
    
    if [ -z "$OCR_FILES" ]; then
        echo "⚠️  No OCR images found on $DEVICE"
        continue
    fi
    
    # Count files
    FILE_COUNT=$(echo "$OCR_FILES" | wc -l | tr -d ' ')
    echo "📊 Found $FILE_COUNT OCR images"
    
    # Pull OCR images
    echo "📥 Downloading..."
    for file in $OCR_FILES; do
        $ADB_CMD pull "$file" "$OUTPUT_DIR/" 2>/dev/null
    done
    
    # Clean up device storage
    echo "🧹 Cleaning up device storage..."
    $ADB_CMD shell "rm -rf /sdcard/Download/onnx_debug"
    
    # Show results for this device
    DOWNLOADED=$(ls "$OUTPUT_DIR"/*_ocr_*.jpg 2>/dev/null | wc -l | tr -d ' ')
    TOTAL_IMAGES=$((TOTAL_IMAGES + DOWNLOADED))
    
    echo "✅ Downloaded $DOWNLOADED images to $OUTPUT_DIR/"
    
    # Show most recent files
    if [ "$DOWNLOADED" -gt 0 ]; then
        echo "📸 Most recent images:"
        ls -lt "$OUTPUT_DIR"/*_ocr_*.jpg 2>/dev/null | head -5 | awk '{print "   " $9}'
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary: Downloaded $TOTAL_IMAGES OCR images total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "OCR Pipeline stages:"
echo "  *_0_ocr_yuv_crop.jpg      - Raw YUV crop from detection bbox"
echo "  *_1_ocr_rgb_converted.jpg - After YUV to RGB conversion"
echo "  *_2_ocr_rotated.jpg       - After 90° CW rotation"
echo "  *_3_ocr_scaled.jpg        - After resize to 640 longest dimension"
echo "  *_4_ocr_final_padded.jpg  - Final 640x640 padded image sent to OCR"
echo ""
echo "🧹 Device storage cleaned up"