#!/usr/bin/env python3
"""
YOLOv8 Automated Pre-labeling Script for Container OCR
=====================================================

This script uses an existing YOLOv8 vertical OCR model to automatically pre-label
container images for training a new model. It's designed specifically for vertical
container codes where characters are arranged vertically with minimal overlap.

Features:
- Batch processing for efficient inference
- Confidence threshold filtering
- Vertical container code specific constraints (max 11 boxes, similar sizes)
- YOLO format output for training
- Quality control with overlapping box filtering
- Progress tracking and logging

Author: AI Assistant
Date: July 2025
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import argparse
from datetime import datetime
import json

# Import config_template for default values
try:
    from config_template import MODEL_PATH, INPUT_DIR, OUTPUT_DIR
except ImportError:
    # If config_template not found, set None as defaults
    MODEL_PATH = None
    INPUT_DIR = None
    OUTPUT_DIR = None

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Please install required packages: pip install ultralytics torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_labeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContainerOCRAutoLabeler:
    """
    Automated pre-labeling system for vertical container OCR using YOLOv8
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.2, 
                 max_boxes: int = 11, min_box_area: float = 50.0):
        """
        Initialize the auto-labeling system

        Args:
            model_path: Path to the trained YOLOv8 model
            confidence_threshold: Minimum confidence score for detections
            max_boxes: Maximum number of boxes allowed per image
            min_box_area: Minimum area (in pixels) for a valid box
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_boxes = max_boxes
        self.min_box_area = min_box_area

        # Load model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'total_boxes': 0,
            'filtered_boxes': 0,
            'avg_confidence': 0.0
        }

    def validate_vertical_container_constraints(self, boxes: np.ndarray, 
                                              confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply vertical container specific constraints and filtering

        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            confidences: Array of confidence scores

        Returns:
            Filtered boxes and confidences
        """
        if len(boxes) == 0:
            return boxes, confidences

        # Filter by confidence threshold
        conf_mask = confidences >= self.confidence_threshold
        boxes = boxes[conf_mask]
        confidences = confidences[conf_mask]

        if len(boxes) == 0:
            return boxes, confidences

        # Calculate box areas and filter by minimum area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_mask = areas >= self.min_box_area
        boxes = boxes[area_mask]
        confidences = confidences[area_mask]
        areas = areas[area_mask]

        if len(boxes) == 0:
            return boxes, confidences

        # Sort by vertical position (y-coordinate) for vertical containers
        y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
        sort_indices = np.argsort(y_centers)
        boxes = boxes[sort_indices]
        confidences = confidences[sort_indices]
        areas = areas[sort_indices]

        # Apply size similarity constraint for container characters
        # Remove boxes that are significantly different in size from the median
        median_area = np.median(areas)
        area_ratio_threshold = 5.0  # Allow 5x variation from median (more lenient)

        area_ratios = areas / median_area
        size_mask = (area_ratios >= 1/area_ratio_threshold) & (area_ratios <= area_ratio_threshold)

        boxes = boxes[size_mask]
        confidences = confidences[size_mask]

        # Apply minimal overlap constraint
        # Remove boxes that have significant overlap with higher confidence boxes
        boxes, confidences = self._remove_overlapping_boxes(boxes, confidences)

        # Limit to maximum number of boxes
        if len(boxes) > self.max_boxes:
            # Keep the highest confidence boxes
            top_indices = np.argsort(confidences)[-self.max_boxes:]
            boxes = boxes[top_indices]
            confidences = confidences[top_indices]

            # Re-sort by vertical position
            y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
            sort_indices = np.argsort(y_centers)
            boxes = boxes[sort_indices]
            confidences = confidences[sort_indices]

        return boxes, confidences

    def _remove_overlapping_boxes(self, boxes: np.ndarray, 
                                 confidences: np.ndarray, 
                                 iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove overlapping boxes using Non-Maximum Suppression

        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            confidences: Array of confidence scores
            iou_threshold: IoU threshold for NMS

        Returns:
            Filtered boxes and confidences
        """
        if len(boxes) == 0:
            return boxes, confidences

        # Calculate IoU between all pairs of boxes
        keep_indices = []
        indices = list(range(len(boxes)))

        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]

        while len(sorted_indices) > 0:
            # Take the box with highest confidence
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_indices = sorted_indices[1:]
            remaining_boxes = boxes[remaining_indices]

            # Calculate IoU
            ious = self._calculate_iou(current_box, remaining_boxes)

            # Keep boxes with IoU less than threshold
            keep_mask = ious < iou_threshold
            sorted_indices = remaining_indices[keep_mask]

        return boxes[keep_indices], confidences[keep_indices]

    def _calculate_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between one box and multiple boxes

        Args:
            box1: Single box [x1, y1, x2, y2]
            boxes: Multiple boxes [N, 4]

        Returns:
            IoU values for each box
        """
        # Calculate intersection areas
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-10)
        return iou

    def process_image(self, image_path: str) -> Optional[List[dict]]:
        """
        Process a single image and return YOLO format annotations

        Args:
            image_path: Path to the image file

        Returns:
            List of annotation dictionaries or None if processing failed
        """
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None

            h, w = image.shape[:2]

            # Run inference
            results = self.model(image_path, verbose=False, conf=0.005)  # Use very low conf for initial detection

            if not results or len(results) == 0:
                logger.warning(f"No detections in image: {image_path}")
                return []

            result = results[0]

            # Extract detections
            if result.boxes is None or len(result.boxes) == 0:
                return []

            # Get bounding boxes and confidences
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            # Apply vertical container constraints
            boxes, confidences = self.validate_vertical_container_constraints(boxes, confidences)

            if len(boxes) == 0:
                return []

            # Convert to YOLO format
            annotations = []
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box

                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # Use class 0 for all detections (assuming single class for characters)
                class_id = 0

                annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': conf
                })

            # Update statistics
            self.stats['total_boxes'] += len(annotations)
            if len(annotations) > 0:
                self.stats['avg_confidence'] = (
                    self.stats['avg_confidence'] * self.stats['processed_images'] + 
                    np.mean([ann['confidence'] for ann in annotations])
                ) / (self.stats['processed_images'] + 1)

            return annotations

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def process_batch(self, image_paths: List[str], 
                     output_dir: str, 
                     batch_size: int = 32) -> None:
        """
        Process a batch of images and save annotations

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save annotation files
            batch_size: Number of images to process at once
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process images with progress bar
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]

                for image_path in batch_paths:
                    self.stats['total_images'] += 1

                    # Process image
                    annotations = self.process_image(image_path)

                    if annotations is not None:
                        # Save annotation file
                        image_name = Path(image_path).stem
                        annotation_file = os.path.join(output_dir, f"{image_name}.txt")

                        self.save_yolo_annotations(annotations, annotation_file)
                        self.stats['processed_images'] += 1
                    else:
                        self.stats['failed_images'] += 1

                    pbar.update(1)

    def save_yolo_annotations(self, annotations: List[dict], output_file: str) -> None:
        """
        Save annotations in YOLO format

        Args:
            annotations: List of annotation dictionaries
            output_file: Path to save the annotation file
        """
        try:
            with open(output_file, 'w') as f:
                for ann in annotations:
                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                    f.write(line)
        except Exception as e:
            logger.error(f"Error saving annotations to {output_file}: {e}")

    def print_statistics(self) -> None:
        """Print processing statistics"""
        logger.info("\n" + "="*50)
        logger.info("AUTO-LABELING STATISTICS")
        logger.info("="*50)
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Successfully processed: {self.stats['processed_images']}")
        logger.info(f"Failed to process: {self.stats['failed_images']}")
        logger.info(f"Success rate: {self.stats['processed_images']/self.stats['total_images']*100:.1f}%")
        logger.info(f"Total boxes generated: {self.stats['total_boxes']}")
        logger.info(f"Average confidence: {self.stats['avg_confidence']:.3f}")
        logger.info(f"Average boxes per image: {self.stats['total_boxes']/max(self.stats['processed_images'], 1):.1f}")
        logger.info("="*50)

    def save_statistics(self, output_file: str) -> None:
        """Save statistics to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")

def get_image_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from directory

    Args:
        input_dir: Directory containing images
        extensions: List of valid image extensions

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    return [str(f) for f in sorted(image_files)]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv8 Automated Pre-labeling for Container OCR")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained YOLOv8 model (.pt file)")
    parser.add_argument("--input", default=INPUT_DIR, help="Directory containing images to label")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory to save YOLO format annotations")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold (default: 0.2)")
    parser.add_argument("--max-boxes", type=int, default=11, help="Maximum boxes per image (default: 11)")
    parser.add_argument("--min-area", type=float, default=50.0, help="Minimum box area in pixels (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("--extensions", nargs='+', default=['.jpg', '.jpeg', '.png'], 
                       help="Image file extensions to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (default: process all)")

    args = parser.parse_args()

    # Validate inputs
    if not args.model:
        logger.error("Model path not specified. Use --model or set MODEL_PATH in config_template.py")
        return
    
    if not args.input:
        logger.error("Input directory not specified. Use --input or set INPUT_DIR in config_template.py")
        return
    
    if not args.output:
        logger.error("Output directory not specified. Use --output or set OUTPUT_DIR in config_template.py")
        return

    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return

    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        return

    # Get image files
    image_files = get_image_files(args.input, args.extensions)

    if not image_files:
        logger.error(f"No image files found in {args.input}")
        return

    logger.info(f"Found {len(image_files)} image files")
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        image_files = image_files[:args.limit]
        logger.info(f"Limiting processing to {len(image_files)} images")

    # Initialize auto-labeler
    auto_labeler = ContainerOCRAutoLabeler(
        model_path=args.model,
        confidence_threshold=args.confidence,
        max_boxes=args.max_boxes,
        min_box_area=args.min_area
    )

    # Process images
    start_time = datetime.now()
    logger.info(f"Starting auto-labeling at {start_time}")

    auto_labeler.process_batch(
        image_paths=image_files,
        output_dir=args.output,
        batch_size=args.batch_size
    )

    end_time = datetime.now()
    processing_time = end_time - start_time

    logger.info(f"Completed auto-labeling in {processing_time}")

    # Print and save statistics
    auto_labeler.print_statistics()
    auto_labeler.save_statistics(os.path.join(args.output, "labeling_stats.json"))

    logger.info(f"Annotations saved to: {args.output}")
    logger.info("Auto-labeling completed successfully!")

if __name__ == "__main__":
    main()
