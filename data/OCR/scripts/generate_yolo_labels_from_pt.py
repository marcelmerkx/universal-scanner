#!/usr/bin/env python3
"""
Generate YOLO Labels from PyTorch Model for Horizontal Containers
=================================================================

This script processes container images (with container codes as filenames)
and generates YOLO format labels using a PyTorch model.

Assumptions:
- Images are named with their container code (e.g., MWCU5304059.jpg)
- Images are cutouts of containers
- Container codes are horizontal, left to right
- Output follows YOLO format: class_id x_center y_center width height (normalized)

Usage:
    python generate_yolo_labels_from_pt.py --input /path/to/images --model /path/to/model.pt
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime
from tqdm import tqdm
import json
import re


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"label_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_container_code(filename):
    """Extract container code from filename"""
    # Remove extension
    name = Path(filename).stem
    
    # Clean up the name - remove spaces, dashes, underscores
    name = re.sub(r'[-_\s]', '', name)
    
    # Extract alphanumeric characters
    code = re.sub(r'[^A-Z0-9]', '', name.upper())
    
    # Validate container code format (4 letters + 7 digits)
    if len(code) == 11 and code[:4].isalpha() and code[4:].isdigit():
        return code
    
    return None


def process_image(image_path, model, logger):
    """Process a single image and generate YOLO labels"""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Run inference - use lower confidence to catch more characters
        results = model(image, imgsz=320, conf=0.3)
        
        # Process detections
        yolo_annotations = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Convert each detection to YOLO format
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Calculate center coordinates and dimensions (normalized)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # YOLO format: class_id x_center y_center width height
                yolo_annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': bbox_width,
                    'height': bbox_height,
                    'confidence': confidence,
                    'x_pixel': (x1 + x2) / 2  # For sorting
                })
        
        # Sort by x-coordinate (left to right)
        yolo_annotations.sort(key=lambda x: x['x_pixel'])
        
        return yolo_annotations
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None


def save_yolo_labels(annotations, output_path):
    """Save annotations in YOLO format"""
    with open(output_path, 'w') as f:
        for ann in annotations:
            # Write YOLO format line
            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                   f"{ann['width']:.6f} {ann['height']:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate YOLO labels from PyTorch model')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--model', type=str, required=True, help='Path to PyTorch model (.pt)')
    parser.add_argument('--output', type=str, help='Output directory for labels (default: input/../labels)')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png'],
                       help='Image file extensions to process')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Default output to sibling directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / 'labels'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check model
    model_path = Path(args.model)
    if not model_path.exists() or model_path.suffix != '.pt':
        print(f"Error: PyTorch model not found or invalid: {model_path}")
        return
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting YOLO label generation")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Confidence threshold: {args.conf}")
    
    # Load model
    logger.info("Loading PyTorch model...")
    model = YOLO(str(model_path))
    
    # Character mapping for logging
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
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
        'total_detections': 0,
        'avg_detections_per_image': 0,
        'validation_results': []
    }
    
    # Process each image
    for image_path in tqdm(image_files, desc="Generating labels"):
        stats['processed'] += 1
        
        # Extract expected code from filename
        expected_code = extract_container_code(image_path.name)
        
        # Process image
        annotations = process_image(image_path, model, logger)
        
        if annotations is None:
            stats['failed'] += 1
            continue
        
        if len(annotations) == 0:
            logger.warning(f"No detections for {image_path.name}")
            stats['failed'] += 1
            continue
        
        # Save YOLO labels
        label_path = output_dir / f"{image_path.stem}.txt"
        save_yolo_labels(annotations, label_path)
        
        stats['successful'] += 1
        stats['total_detections'] += len(annotations)
        
        # Validate against filename if possible
        if expected_code:
            detected_chars = []
            for ann in annotations:
                if ann['class_id'] < len(char_map):
                    detected_chars.append(char_map[ann['class_id']])
            
            detected_code = ''.join(detected_chars)
            is_match = detected_code == expected_code
            
            validation_result = {
                'filename': image_path.name,
                'expected': expected_code,
                'detected': detected_code,
                'match': is_match,
                'num_detections': len(annotations),
                'avg_confidence': sum(a['confidence'] for a in annotations) / len(annotations)
            }
            stats['validation_results'].append(validation_result)
            
            if not is_match:
                logger.debug(f"Mismatch: {image_path.name} - Expected: {expected_code}, "
                           f"Detected: {detected_code}")
    
    # Calculate final statistics
    if stats['successful'] > 0:
        stats['avg_detections_per_image'] = stats['total_detections'] / stats['successful']
    
    # Save statistics
    stats_file = output_dir / "generation_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Calculate validation accuracy if available
    if stats['validation_results']:
        matches = sum(1 for v in stats['validation_results'] if v['match'])
        accuracy = (matches / len(stats['validation_results'])) * 100
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("LABEL GENERATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Successfully processed: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Average detections per image: {stats['avg_detections_per_image']:.1f}")
    
    if stats['validation_results']:
        logger.info(f"\nValidation Results:")
        logger.info(f"Validated: {len(stats['validation_results'])} images")
        logger.info(f"Matches: {matches}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  Labels: {output_dir}/*.txt")
    logger.info(f"  Statistics: {stats_file}")
    logger.info(f"  Log: {output_dir}/*.log")


if __name__ == "__main__":
    main()