#!/usr/bin/env python3
"""
YOLOv8 Automated Pre-labeling Script for Horizontal Container OCR
==================================================================

This script uses a YOLOv8 OCR model to automatically pre-label horizontal container
images for training. It's designed specifically for horizontal container codes where
characters are arranged left-to-right.

Features:
- Processes horizontal container images
- Validates detected codes against filename (first 11 characters)
- Separates correctly labeled images from errors
- Handles both single-line and potentially multi-line containers
- Generates YOLO format labels
- Comprehensive error tracking and reporting

Usage:
    # Basic usage
    python horizontal_container_auto_labeler.py --input /path/to/images --output /path/to/output
    
    # With specific model and confidence
    python horizontal_container_auto_labeler.py \
        --input /path/to/images \
        --output /path/to/output \
        --model data/OCR/models/container-ocr-bt9wl-v3-21072025.pt \
        --confidence 0.4

Output structure:
    output/
    ├── labels/         # Correctly matched labels
    ├── images/         # Correctly matched images (copied)
    ├── error/          # Mismatched cases
    │   ├── labels/     # Mismatched labels
    │   └── images/     # Mismatched images
    └── labeling_report.json
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import shutil
import re

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Please install required packages: pip install ultralytics torch")
    sys.exit(1)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file = output_dir / f"horizontal_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_container_code_from_filename(filename: str) -> Optional[str]:
    """
    Extract container code from filename (first 11 characters).
    
    Container code format: 4 letters + 7 digits (ISO 6346)
    """
    # Remove extension
    name = Path(filename).stem
    
    # Get first 11 characters
    if len(name) >= 11:
        potential_code = name[:11].upper()
        
        # Validate format: 4 letters + 7 digits
        if re.match(r'^[A-Z]{4}\d{7}$', potential_code):
            return potential_code
    
    return None


def validate_container_code(code: str) -> bool:
    """Validate container code format"""
    # Must be 11 characters: 4 letters + 7 digits
    if len(code) != 11:
        return False
    
    # Check format
    if not re.match(r'^[A-Z]{4}\d{7}$', code):
        return False
    
    # Additional checks could be added here (e.g., check digit validation)
    return True


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def apply_nms(detections: List[Dict], iou_threshold: float = 0.2) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression (0.2 = 80% overlap)
    
    Returns:
        Filtered list of detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (descending)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, det1 in enumerate(sorted_detections):
        if i in suppressed:
            continue
            
        keep.append(det1)
        
        # Check against remaining detections
        for j in range(i + 1, len(sorted_detections)):
            if j in suppressed:
                continue
                
            det2 = sorted_detections[j]
            iou = calculate_iou(det1['bbox'], det2['bbox'])
            
            # If IoU > threshold (80% overlap when threshold=0.2), suppress the lower confidence one
            if iou > iou_threshold:
                suppressed.add(j)
    
    return keep


def process_image(image_path: Path, model, conf_threshold: float = 0.4) -> Tuple[List[Dict], np.ndarray]:
    """
    Process a single image and return detections.
    
    Returns:
        Tuple of (detections_list, image_shape)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Run inference
    results = model(image, imgsz=320, conf=conf_threshold, verbose=False)
    
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            
            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': class_id,
                'confidence': confidence,
                'x_center': (x1 + x2) / 2,
                'y_center': (y1 + y2) / 2
            }
            detections.append(detection)
    
    return detections, (height, width)


def detections_to_text(detections: List[Dict], char_map: str) -> str:
    """Convert detections to text string by sorting left to right"""
    # Sort by x-coordinate (left to right)
    sorted_detections = sorted(detections, key=lambda d: d['x_center'])
    
    text = ""
    for det in sorted_detections:
        if det['class_id'] < len(char_map):
            text += char_map[det['class_id']]
    
    return text


def filter_detections_for_horizontal_container(detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
    """
    Filter detections to get the main horizontal container code.
    
    For horizontal containers:
    - Characters should be roughly aligned horizontally
    - Should have around 11 characters
    - Remove outliers that are too far from the main line
    """
    if len(detections) < 11:
        return detections
    
    # Sort by x-coordinate
    sorted_detections = sorted(detections, key=lambda d: d['x_center'])
    
    # If we have exactly 11 or close, return them
    if 11 <= len(sorted_detections) <= 13:
        return sorted_detections[:11]
    
    # Find the main horizontal line by clustering y-coordinates
    y_centers = [d['y_center'] for d in sorted_detections]
    median_y = np.median(y_centers)
    
    # Filter detections that are close to the median y
    height = image_shape[0]
    y_tolerance = height * 0.1  # 10% of image height
    
    filtered = [d for d in sorted_detections 
                if abs(d['y_center'] - median_y) < y_tolerance]
    
    # Take the first 11 characters
    return filtered[:11]


def save_yolo_labels(detections: List[Dict], image_shape: Tuple[int, int], output_path: Path):
    """Save detections in YOLO format"""
    height, width = image_shape
    
    with open(output_path, 'w') as f:
        for det in detections:
            # Calculate normalized coordinates
            x_center = ((det['bbox'][0] + det['bbox'][2]) / 2) / width
            y_center = ((det['bbox'][1] + det['bbox'][3]) / 2) / height
            bbox_width = (det['bbox'][2] - det['bbox'][0]) / width
            bbox_height = (det['bbox'][3] - det['bbox'][1]) / height
            
            # Write YOLO format line
            f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} "
                   f"{bbox_width:.6f} {bbox_height:.6f}\n")


def process_batch(image_paths: List[Path], model, output_dir: Path, 
                 conf_threshold: float, nms_threshold: float, char_map: str, logger: logging.Logger) -> Dict:
    """Process a batch of images"""
    
    # Create output directories
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    error_dir = output_dir / "error"
    error_labels_dir = error_dir / "labels"
    error_images_dir = error_dir / "images"
    
    for dir_path in [labels_dir, images_dir, error_labels_dir, error_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total_images': len(image_paths),
        'processed': 0,
        'successful_matches': 0,
        'mismatches': 0,
        'no_filename_code': 0,
        'no_detections': 0,
        'errors': 0,
        'mismatch_details': []
    }
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Extract expected code from filename
            expected_code = extract_container_code_from_filename(image_path.name)
            
            if not expected_code:
                logger.warning(f"No valid container code in filename: {image_path.name}")
                stats['no_filename_code'] += 1
                continue
            
            # Process image
            detections, image_shape = process_image(image_path, model, conf_threshold)
            
            if len(detections) == 0:
                logger.warning(f"No detections for {image_path.name}")
                stats['no_detections'] += 1
                continue
            
            # Apply NMS to remove overlapping boxes (keep highest confidence)
            detections = apply_nms(detections, iou_threshold=nms_threshold)
            
            # Filter detections for horizontal container
            filtered_detections = filter_detections_for_horizontal_container(detections, image_shape)
            
            # Convert to text
            detected_code = detections_to_text(filtered_detections, char_map)
            
            # Determine if it's a match
            is_match = detected_code == expected_code
            
            # Save based on match status
            if is_match:
                # Save to correct labels/images
                label_path = labels_dir / f"{image_path.stem}.txt"
                save_yolo_labels(filtered_detections, image_shape, label_path)
                
                # Copy image
                shutil.copy2(image_path, images_dir / image_path.name)
                
                stats['successful_matches'] += 1
                logger.debug(f"Match: {image_path.name} - {detected_code}")
                
            else:
                # Save to error folders
                label_path = error_labels_dir / f"{image_path.stem}.txt"
                save_yolo_labels(filtered_detections, image_shape, label_path)
                
                # Copy image
                shutil.copy2(image_path, error_images_dir / image_path.name)
                
                stats['mismatches'] += 1
                
                mismatch_detail = {
                    'filename': image_path.name,
                    'expected': expected_code,
                    'detected': detected_code,
                    'num_detections': len(filtered_detections),
                    'all_detections': len(detections)
                }
                stats['mismatch_details'].append(mismatch_detail)
                
                logger.warning(f"Mismatch: {image_path.name} - Expected: {expected_code}, "
                             f"Detected: {detected_code}")
            
            stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            stats['errors'] += 1
            continue
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Auto-label horizontal container images')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--model', type=str, 
                       default='data/OCR/models/container-ocr-bt9wl-v3-21072025.pt',
                       help='Path to YOLO model')
    parser.add_argument('--confidence', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.2, 
                       help='NMS IoU threshold (0.2 = remove boxes with >80% overlap)')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png'],
                       help='Image file extensions')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    model_path = Path(args.model)
    
    # Validate paths
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("="*50)
    logger.info("Horizontal Container Auto-Labeler")
    logger.info("="*50)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"NMS threshold: {args.nms_threshold} (removes boxes with >{(1-args.nms_threshold)*100:.0f}% overlap)")
    
    # Character mapping - matches training data
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Load model
    logger.info("Loading YOLO model...")
    model = YOLO(str(model_path))
    
    # Collect image files
    image_files = []
    for ext in args.extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    # Apply limit if specified
    if args.limit:
        image_files = image_files[:args.limit]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process in batches
    all_stats = process_batch(image_files, model, output_dir, 
                             args.confidence, args.nms_threshold, char_map, logger)
    
    # Save detailed report
    report_path = output_dir / "labeling_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("LABELING COMPLETE")
    logger.info("="*50)
    logger.info(f"Total images: {all_stats['total_images']}")
    logger.info(f"Processed: {all_stats['processed']}")
    logger.info(f"Successful matches: {all_stats['successful_matches']}")
    logger.info(f"Mismatches: {all_stats['mismatches']}")
    logger.info(f"No filename code: {all_stats['no_filename_code']}")
    logger.info(f"No detections: {all_stats['no_detections']}")
    logger.info(f"Errors: {all_stats['errors']}")
    
    if all_stats['processed'] > 0:
        accuracy = (all_stats['successful_matches'] / all_stats['processed']) * 100
        logger.info(f"\nAccuracy: {accuracy:.2f}%")
    
    logger.info(f"\nOutput structure:")
    logger.info(f"  Correct labels: {output_dir}/labels/")
    logger.info(f"  Correct images: {output_dir}/images/")
    logger.info(f"  Error labels: {output_dir}/error/labels/")
    logger.info(f"  Error images: {output_dir}/error/images/")
    logger.info(f"  Report: {report_path}")


if __name__ == "__main__":
    main()