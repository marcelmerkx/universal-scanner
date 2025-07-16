#!/usr/bin/env python3
"""
Re-label all character images using the character detection model.

This script:
1. Finds all images in data/detection/training_data/02_characters/images
2. Runs inference on each image using the character detection model
3. Creates/overwrites YOLO labels in data/detection/training_data/02_characters/labels

Usage:
    python3 data/detection/scripts/relabel_character_images.py
    python3 data/detection/scripts/relabel_character_images.py --confidence 0.3
    python3 data/detection/scripts/relabel_character_images.py --limit 10
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed. Run: pip install ultralytics")
    exit(1)

# Configuration
CHARACTERS_DIR = Path("data/detection/training_data/11_characters_extra")
DEFAULT_MODEL_PATH = Path("data/detection/models/character-detection-10-06-25.pt")
LOG_DIR = Path("data/detection/logs")


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "relabel_character_images.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_yolo_model(model_path: Path, logger: logging.Logger):
    """
    Load local YOLO model.
    
    Args:
        model_path: Path to the .pt model file
        logger: Logger instance
        
    Returns:
        YOLO model instance
    """
    try:
        logger.info(f"Loading YOLO model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(str(model_path))
        logger.info("✅ Model loaded successfully")
        
        # Print model info
        logger.info(f"Model classes: {list(model.names.values())}")
        logger.info(f"Number of classes: {len(model.names)}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def convert_to_yolo_format(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from xyxy format to YOLO format (normalized center x, y, width, height).
    
    Args:
        bbox: [x1, y1, x2, y2] in pixels
        img_width: Image width
        img_height: Image height
        
    Returns:
        (center_x, center_y, width, height) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center point
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Normalize by image dimensions
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    # Ensure values are in [0, 1] range
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height


def preprocess_image_for_character_model(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Preprocess image for character detection model.
    - Keep color (no grayscale conversion)
    - Resize to 640px wide
    - Add white padding at bottom to make 640x640
    
    Args:
        image: Original BGR image array
        target_size: Target size (640)
        
    Returns:
        Preprocessed image
    """
    height, width = image.shape[:2]
    
    # Calculate scale to make width = 640
    scale = target_size / width
    
    # Calculate new dimensions
    new_width = target_size
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create padded image with white background
    padded = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    
    # Place resized image at top (no centering, just pad bottom)
    padded[:new_height, :new_width] = resized
    
    return padded


def process_images(model, images_dir: Path, labels_dir: Path, 
                  confidence_threshold: float, limit: int, logger: logging.Logger) -> Dict:
    """
    Process all images and create YOLO labels.
    
    Args:
        model: Loaded YOLO model
        images_dir: Directory containing images
        labels_dir: Directory where labels will be saved
        confidence_threshold: Minimum confidence for detections
        limit: Maximum number of images to process (None for all)
        logger: Logger instance
        
    Returns:
        Dictionary with processing statistics
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([f for f in images_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions])
    
    if limit:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create labels directory if it doesn't exist
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        "total_images": len(image_files),
        "images_processed": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "failed_images": 0,
        "processing_time": 0.0,
        "detections_per_class": {}
    }
    
    start_time = time.time()
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        
        try:
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                stats["failed_images"] += 1
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Preprocess image for character model
            preprocessed = preprocess_image_for_character_model(image)
            
            # Save preprocessed image temporarily for inference
            temp_path = f"/tmp/preprocessed_char_{image_path.stem}.jpg"
            cv2.imwrite(temp_path, preprocessed)
            
            # Run inference on preprocessed image
            results = model(temp_path, conf=confidence_threshold, verbose=False)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            # Process detections
            label_lines = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Calculate scale factor used in preprocessing
                scale = 640.0 / img_width
                
                # Collect all detections with their confidence scores
                detections = []
                for i in range(len(boxes)):
                    # Get detection info (coordinates are in 640x640 space)
                    x1_640, y1_640, x2_640, y2_640 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Convert coordinates back to original image space
                    x1_orig = x1_640 / scale
                    y1_orig = y1_640 / scale
                    x2_orig = x2_640 / scale
                    y2_orig = y2_640 / scale
                    
                    # Convert to YOLO format
                    bbox = [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)]
                    center_x, center_y, width, height = convert_to_yolo_format(
                        bbox, img_width, img_height
                    )
                    
                    # Store detection with confidence for sorting
                    detections.append({
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'bbox': bbox
                    })
                
                # Sort by confidence (highest first) and take top 11
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                top_detections = detections[:11]  # Take only top 11
                
                # Create label lines for top detections
                for det in top_detections:
                    label_line = f"{det['class_id']} {det['center_x']:.6f} {det['center_y']:.6f} {det['width']:.6f} {det['height']:.6f}"
                    label_lines.append(label_line)
                    
                    # Update statistics for kept detections
                    stats["total_detections"] += 1
                    if det['class_name'] not in stats["detections_per_class"]:
                        stats["detections_per_class"][det['class_name']] = 0
                    stats["detections_per_class"][det['class_name']] += 1
                    
                    logger.debug(f"  Detection: {det['class_name']} (class {det['class_id']}), "
                               f"conf={det['confidence']:.3f}, bbox={det['bbox']}")
                
                # Log filtering info
                if len(detections) > 11:
                    logger.info(f"  Filtered from {len(detections)} to {len(top_detections)} detections (top 11 by confidence)")
                    logger.info(f"  Confidence range kept: {top_detections[-1]['confidence']:.3f} to {top_detections[0]['confidence']:.3f}")
                elif len(detections) > 0:
                    logger.debug(f"  Kept all {len(detections)} detections (≤11 found)")
            
            # Write labels to file ONLY if we have actual detections
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_lines:
                with open(label_path, 'w') as f:
                    for line in label_lines:
                        f.write(line + '\n')
                stats["images_with_detections"] += 1
                logger.debug(f"  Wrote {len(label_lines)} labels to {label_path}")
            else:
                # If no detections, create empty label file or delete existing one
                if label_path.exists():
                    label_path.unlink()
                logger.debug(f"  No detections found for {image_path.name}")
            
            stats["images_processed"] += 1
            
            # Progress update
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                logger.info(f"Progress: {idx}/{len(image_files)} images "
                          f"({rate:.1f} images/sec)")
                
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            stats["failed_images"] += 1
            continue
    
    stats["processing_time"] = time.time() - start_time
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print processing statistics."""
    logger.info("\n=== PROCESSING STATISTICS ===")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Images processed: {stats['images_processed']}")
    logger.info(f"Failed images: {stats['failed_images']}")
    logger.info(f"Images with detections: {stats['images_with_detections']}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Processing time: {stats['processing_time']:.1f}s")
    
    if stats['images_processed'] > 0:
        rate = stats['images_processed'] / stats['processing_time']
        logger.info(f"Processing rate: {rate:.1f} images/sec")
        
        avg_detections = stats['total_detections'] / stats['images_processed']
        logger.info(f"Average detections per image: {avg_detections:.1f}")
        
        detection_rate = stats['images_with_detections'] / stats['images_processed'] * 100
        logger.info(f"Detection rate: {detection_rate:.1f}%")
    
    logger.info("\nDetections per class:")
    for class_name, count in sorted(stats['detections_per_class'].items()):
        logger.info(f"  {class_name}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-label character images using character detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO character detection model (.pt file)"
    )
    
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=CHARACTERS_DIR / "images",
        help="Directory containing character images"
    )
    
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=CHARACTERS_DIR / "labels",
        help="Directory where labels will be saved"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Character Image Re-labeling Script ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Labels directory: {args.labels_dir}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Image limit: {args.limit if args.limit else 'None (process all)'}")
    
    # Validate paths
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    if not args.images_dir.exists():
        logger.error(f"Images directory not found: {args.images_dir}")
        return 1
    
    try:
        # Load model
        model = load_yolo_model(args.model_path, logger)
        
        # Process images
        logger.info("\nStarting image processing...")
        stats = process_images(
            model, args.images_dir, args.labels_dir,
            args.confidence, args.limit, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        logger.info("\n✅ Re-labeling completed successfully!")
        logger.info(f"Labels saved to: {args.labels_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())