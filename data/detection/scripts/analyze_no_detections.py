#!/usr/bin/env python3
"""
Analyze confidence levels for images with no detections above threshold.

This script:
1. Loads images from the no_detection folder
2. Runs inference with very low confidence threshold
3. Reports all detections and their confidence levels
4. Helps understand why images didn't pass the main threshold

Usage:
    python3 data/detection/scripts/analyze_no_detections.py
    python3 data/detection/scripts/analyze_no_detections.py --confidence 0.01
    python3 data/detection/scripts/analyze_no_detections.py --show-top 5
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed. Run: pip install ultralytics")
    exit(1)

# Configuration
DEFAULT_MODEL_PATH = Path("data/detection/models/detection_320_grayscale_tilted-09-07-2025.pt")
NO_DETECTION_DIR = Path("data/detection/training_data/00_raw/no_detection/images")

# Suppress ultralytics banner
os.environ['SUPRESS_ULTRALYTICS_BANNER'] = '1'


def preprocess_image_for_model(image: np.ndarray, target_size: int = 320) -> Tuple[np.ndarray, float, int, int]:
    """
    Preprocess image to 320x320 grayscale with aspect-ratio preserving padding.
    
    Args:
        image: Original BGR image array
        target_size: Target square size (default 320)
        
    Returns:
        Tuple of (preprocessed_image, scale_factor, x_offset, y_offset)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    
    # Calculate scale to fit within target_size while preserving aspect ratio
    scale = min(target_size / width, target_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(gray, (new_width, new_height))
    
    # Create padded image with white background
    padded = np.full((target_size, target_size), 255, dtype=np.uint8)
    
    # Calculate padding offsets (center the image)
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Convert back to 3-channel for YOLO
    preprocessed = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    
    return preprocessed, scale, x_offset, y_offset


def analyze_image_detections(model, image_path: Path, confidence_threshold: float) -> List[Dict]:
    """
    Analyze detections for a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to image file
        confidence_threshold: Minimum confidence for analysis
        
    Returns:
        List of detection dictionaries
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return []
    
    # Preprocess image
    preprocessed, scale, x_offset, y_offset = preprocess_image_for_model(image)
    
    # Save preprocessed image temporarily for inference
    temp_path = f"/tmp/preprocessed_{image_path.stem}.jpg"
    cv2.imwrite(temp_path, preprocessed)
    
    # Run inference with low confidence
    results = model(temp_path, conf=confidence_threshold, verbose=False)
    
    # Clean up temp file
    Path(temp_path).unlink(missing_ok=True)
    
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = model.names[class_id]
            
            detection = {
                'class_name': class_name,
                'confidence': confidence,
                'bbox_320': [int(x1), int(y1), int(x2), int(y2)],
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'area': int((x2 - x1) * (y2 - y1))
            }
            detections.append(detection)
    
    return detections


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze confidence levels for no-detection images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO model (.pt file)"
    )
    
    parser.add_argument(
        "--no-detection-dir",
        type=Path,
        default=NO_DETECTION_DIR,
        help="Directory containing no-detection images"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.01,
        help="Minimum confidence threshold for analysis"
    )
    
    parser.add_argument(
        "--show-top",
        type=int,
        default=None,
        help="Show only top N detections per image (by confidence)"
    )
    
    parser.add_argument(
        "--sort-by",
        choices=['confidence', 'filename'],
        default='filename',
        help="Sort results by confidence or filename"
    )
    
    args = parser.parse_args()
    
    print("=== No-Detection Image Analysis ===")
    print(f"Model: {args.model_path}")
    print(f"Input directory: {args.no_detection_dir}")
    print(f"Analysis confidence: {args.confidence}")
    print(f"Show top detections: {args.show_top if args.show_top else 'All'}")
    print("="*50)
    
    # Validate paths
    if not args.model_path.exists():
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    if not args.no_detection_dir.exists():
        print(f"❌ No-detection directory not found: {args.no_detection_dir}")
        print("Run inference_and_crop.py first to generate no-detection images")
        return 1
    
    # Load model
    print("Loading YOLO model...")
    model = YOLO(str(args.model_path))
    print(f"✅ Model loaded. Classes: {list(model.names.values())}")
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in args.no_detection_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {args.no_detection_dir}")
        return 1
    
    print(f"\nFound {len(image_files)} images to analyze")
    print("="*50)
    
    # Analyze each image
    all_results = []
    
    for image_path in image_files:
        print(f"\n📸 {image_path.name}")
        print("-" * 40)
        
        detections = analyze_image_detections(model, image_path, args.confidence)
        
        if detections:
            # Sort detections by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit to top N if specified
            if args.show_top:
                detections = detections[:args.show_top]
            
            print(f"Found {len(detections)} detection(s):")
            
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']:15s} conf={det['confidence']:.4f} "
                      f"bbox={det['bbox_320']} size={det['width']}x{det['height']}")
                
                # Store for summary
                result = {
                    'filename': image_path.name,
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'bbox': det['bbox_320'],
                    'size': f"{det['width']}x{det['height']}"
                }
                all_results.append(result)
        else:
            print(f"  ❌ No detections found (even at {args.confidence} confidence)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_results:
        # Group by confidence ranges
        confidence_ranges = {
            '0.40-0.50': [r for r in all_results if 0.40 <= r['confidence'] < 0.50],
            '0.30-0.40': [r for r in all_results if 0.30 <= r['confidence'] < 0.40],
            '0.20-0.30': [r for r in all_results if 0.20 <= r['confidence'] < 0.30],
            '0.10-0.20': [r for r in all_results if 0.10 <= r['confidence'] < 0.20],
            '0.01-0.10': [r for r in all_results if 0.01 <= r['confidence'] < 0.10]
        }
        
        print(f"Total detections found: {len(all_results)}")
        print(f"Images with some detections: {len(set(r['filename'] for r in all_results))}")
        print(f"Images with no detections: {len(image_files) - len(set(r['filename'] for r in all_results))}")
        
        print(f"\nConfidence distribution:")
        for range_name, results in confidence_ranges.items():
            if results:
                print(f"  {range_name}: {len(results)} detections")
        
        # Class distribution
        print(f"\nClass distribution:")
        class_counts = {}
        for result in all_results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            avg_conf = np.mean([r['confidence'] for r in all_results if r['class_name'] == class_name])
            print(f"  {class_name}: {count} detections (avg conf: {avg_conf:.3f})")
        
        # Show highest confidence detections that didn't make the cut
        print(f"\nTop 10 detections that missed 0.5 threshold:")
        high_conf_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)[:10]
        for i, result in enumerate(high_conf_results, 1):
            print(f"  {i:2d}. {result['filename']:20s} {result['class_name']:15s} "
                  f"conf={result['confidence']:.4f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if any(r['confidence'] >= 0.45 for r in all_results):
            print("✓ Some detections are very close to 0.5 threshold")
            print("  → Consider lowering threshold to 0.4 or 0.45")
        
        if any(r['confidence'] >= 0.3 for r in all_results):
            close_count = len([r for r in all_results if r['confidence'] >= 0.3])
            print(f"✓ {close_count} detections above 0.3 confidence")
            print("  → These might be valid but need more training")
        
        truly_no_detection = len(image_files) - len(set(r['filename'] for r in all_results))
        if truly_no_detection > 0:
            print(f"⚠️  {truly_no_detection} images have NO detections even at {args.confidence}")
            print("  → These may need manual review or different preprocessing")
    
    else:
        print(f"❌ No detections found in any image at {args.confidence} confidence")
        print("  → Try lowering the confidence threshold (e.g., --confidence 0.001)")
    
    return 0


if __name__ == "__main__":
    exit(main())