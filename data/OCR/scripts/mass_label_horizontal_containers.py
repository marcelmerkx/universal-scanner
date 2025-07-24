#!/usr/bin/env python3
"""
Mass Labeling Pipeline for Horizontal Container Images
======================================================

This script processes many container code cutouts, resizes them to 320x320,
applies grayscale conversion, runs inference, and generates YOLO format labels.
It validates detected codes against the filename (which should be the container code).

Features:
- Batch processing of container images
- Grayscale conversion for consistency
- Automatic validation against filename
- YOLO format label generation
- Comprehensive logging of differences
- Progress tracking

Usage:
    python mass_label_horizontal_containers.py --input /path/to/images --output /path/to/labels --model /path/to/model.onnx
"""

import os
import cv2
import numpy as np
import argparse
import onnxruntime as ort
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import re
from typing import List, Dict, Tuple, Optional


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"mass_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_container_code_from_filename(filename):
    """Extract container code from filename"""
    # Remove extension
    name = Path(filename).stem
    
    # Remove common suffixes/prefixes
    name = re.sub(r'[-_\s]', '', name)
    
    # Extract alphanumeric characters
    code = re.sub(r'[^A-Z0-9]', '', name.upper())
    
    # Validate container code format (4 letters + 7 digits)
    if len(code) == 11 and code[:4].isalpha() and code[4:].isdigit():
        return code
    
    return None


def letterbox_image(image, target_size=(320, 320)):
    """Resize image with padding to maintain aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (gray background)
    if len(resized.shape) == 2:
        padded = np.full((target_h, target_w), 114, dtype=np.uint8)
    else:
        padded = np.full((target_h, target_w, resized.shape[2]), 114, dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, (pad_w, pad_h)


def preprocess_image(image, grayscale=True):
    """Preprocess image for ONNX model"""
    # Convert to grayscale if requested
    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale to RGB format for model
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def process_yolo_output(predictions, confidence_threshold=0.3, input_size=320):
    """Process YOLO output with proper format handling"""
    # Remove batch dimension if present
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    
    shape = predictions.shape
    num_attributes = 40  # 4 bbox + 36 classes
    num_classes = 36
    
    # Determine layout
    if shape[0] == num_attributes:
        # [40, N] layout - attributes first
        attributes_first = True
        num_anchors = shape[1]
    elif shape[1] == num_attributes:
        # [N, 40] layout - anchors first
        attributes_first = False
        num_anchors = shape[0]
    else:
        return [], [], []
    
    boxes = []
    scores = []
    classes = []
    
    # Process each anchor
    for i in range(num_anchors):
        # Get max class probability
        max_prob = 0
        max_class = 0
        
        for c in range(num_classes):
            if attributes_first:
                prob = predictions[4 + c, i]  # [40, N] layout
            else:
                prob = predictions[i, 4 + c]  # [N, 40] layout
            
            if prob > max_prob:
                max_prob = prob
                max_class = c
        
        if max_prob < confidence_threshold:
            continue
        
        # Extract bbox coordinates
        if attributes_first:
            x = predictions[0, i]  # x-center
            y = predictions[1, i]  # y-center
            w = predictions[2, i]  # width
            h = predictions[3, i]  # height
        else:
            x = predictions[i, 0]
            y = predictions[i, 1]
            w = predictions[i, 2]
            h = predictions[i, 3]
        
        # Check if coordinates are normalized (0-1) or pixel space
        if max(x, y, w, h) <= 1.0:
            # Convert from normalized to pixel coordinates
            x = x * input_size
            y = y * input_size
            w = w * input_size
            h = h * input_size
        
        # Convert to corner format
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        boxes.append([x1, y1, x2, y2])
        scores.append(max_prob)
        classes.append(max_class)
    
    return boxes, scores, classes


def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    """Apply Non-Maximum Suppression"""
    if not boxes:
        return [], [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # Sort by confidence
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU
        current_box = boxes[i]
        remaining_boxes = boxes[indices[1:]]
        
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                         (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = box_area + remaining_areas - intersection
        
        iou = intersection / (union + 1e-6)
        
        indices = indices[1:][iou < iou_threshold]
    
    return boxes[keep], scores[keep], classes[keep]


def process_single_image(image_path, session, char_map, logger):
    """Process a single image and return detections"""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None, None
        
        original_shape = image.shape[:2]
        
        # Letterbox resize
        resized, scale, (pad_w, pad_h) = letterbox_image(image, (320, 320))
        
        # Preprocess with grayscale
        preprocessed = preprocess_image(resized, grayscale=True)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: preprocessed})
        predictions = outputs[0]
        
        # Remove batch dimension if present
        if len(predictions.shape) == 3:
            predictions = predictions[0]
        
        # Process YOLO output
        boxes, scores, classes = process_yolo_output(predictions)
        
        if len(boxes) == 0:
            return [], original_shape
        
        # Apply NMS
        boxes, scores, classes = apply_nms(boxes, scores, classes)
        
        # Convert coordinates back to original image space
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            
            # Adjust for padding and scale (coordinates are already in pixel space)
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            # Clip to image bounds
            h, w = original_shape
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Store detection
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': int(cls),
                'confidence': float(score),
                'center_x': (x1 + x2) / 2,
                'char': char_map[int(cls)] if int(cls) < len(char_map) else '?'
            })
        
        # Sort by x-coordinate (left to right for horizontal containers)
        detections = sorted(detections, key=lambda d: d['center_x'])
        
        return detections, original_shape
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None, None


def save_yolo_labels(detections, image_shape, output_path):
    """Save detections in YOLO format"""
    h, w = image_shape
    
    with open(output_path, 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Write in YOLO format: class_id x_center y_center width height
            f.write(f"{det['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Mass label horizontal container images')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for labels')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    model_path = Path(args.model)
    
    # Validate paths
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting mass labeling pipeline")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_path}")
    
    # Character mapping - must match training data!
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Load ONNX model
    logger.info("Loading ONNX model...")
    session = ort.InferenceSession(str(model_path))
    
    # Collect all image files
    image_files = []
    for ext in args.extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Statistics
    stats = {
        'total_images': len(image_files),
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'matches': 0,
        'mismatches': 0,
        'no_filename_code': 0,
        'differences': []
    }
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        stats['processed'] += 1
        
        # Extract expected code from filename
        expected_code = extract_container_code_from_filename(image_path.name)
        if not expected_code:
            logger.warning(f"Could not extract container code from filename: {image_path.name}")
            stats['no_filename_code'] += 1
        
        # Process image
        detections, image_shape = process_single_image(image_path, session, char_map, logger)
        
        if detections is None:
            stats['failed'] += 1
            continue
        
        if len(detections) == 0:
            logger.warning(f"No detections for {image_path.name}")
            stats['failed'] += 1
            continue
        
        # Save YOLO labels
        label_path = output_dir / f"{image_path.stem}.txt"
        save_yolo_labels(detections, image_shape, label_path)
        stats['successful'] += 1
        
        # Build detected code string
        detected_code = ''.join([d['char'] for d in detections])
        
        # Compare with expected code
        if expected_code:
            if detected_code == expected_code:
                stats['matches'] += 1
            else:
                stats['mismatches'] += 1
                difference = {
                    'filename': image_path.name,
                    'expected': expected_code,
                    'detected': detected_code,
                    'num_detections': len(detections),
                    'confidences': [d['confidence'] for d in detections]
                }
                stats['differences'].append(difference)
                logger.warning(f"Mismatch: {image_path.name} - Expected: {expected_code}, Detected: {detected_code}")
        
        # Log progress every 100 images
        if stats['processed'] % 100 == 0:
            logger.info(f"Progress: {stats['processed']}/{stats['total_images']} images processed")
    
    # Save statistics
    stats_file = output_dir / "labeling_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save differences report
    if stats['differences']:
        diff_file = output_dir / "differences_report.json"
        with open(diff_file, 'w') as f:
            json.dump(stats['differences'], f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("LABELING COMPLETE")
    logger.info("="*50)
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Successfully processed: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"No filename code: {stats['no_filename_code']}")
    
    if expected_code:
        logger.info(f"\nValidation Results:")
        logger.info(f"Matches: {stats['matches']}")
        logger.info(f"Mismatches: {stats['mismatches']}")
        if stats['successful'] > 0:
            accuracy = (stats['matches'] / (stats['matches'] + stats['mismatches'])) * 100
            logger.info(f"Accuracy: {accuracy:.2f}%")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  Labels: {output_dir}/*.txt")
    logger.info(f"  Statistics: {stats_file}")
    if stats['differences']:
        logger.info(f"  Differences: {diff_file}")


if __name__ == "__main__":
    main()