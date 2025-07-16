#!/usr/bin/env python3
"""
Re-label all images in the 01_labelled folder for their DETECTIONS (6 classes).

This script:
1. Finds all images in data/detection/training_data/01_labelled/images
2. Runs inference on each image
3. Creates/overwrites YOLO labels in data/detection/training_data/01_labelled/labels

Usage:
    python3 data/detection/scripts/relabel_all_images.py
    python3 data/detection/scripts/relabel_all_images.py --confidence 0.3
    python3 data/detection/scripts/relabel_all_images.py --use-full-image
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
LABELLED_DIR = Path("data/detection/training_data/01_labelled")
# LABELLED_DIR = Path("data/detection/containerdoors")
DEFAULT_MODEL_PATH = Path("data/detection/models/detection_320_grayscale_tilted-09-07-2025.pt")
LOG_DIR = Path("data/detection/logs")

# Class mapping from unified-detection-v7.yaml
CLASS_NAMES_TO_INDEX = {
    'code_container_h': 0,
    'code_container_v': 1,
    'code_license_plate': 2,
    'code_qr_barcode': 3,
    'code_seal': 4,
}


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "relabel_all_images.log"
    
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
    """Load YOLO model for inference."""
    try:
        logger.info(f"Loading YOLO model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(str(model_path))
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model classes: {list(model.names.values())}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def convert_to_yolo_format(bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates to YOLO normalized format.
    
    Args:
        bbox: [x1, y1, x2, y2] in pixels
        img_width: Image width
        img_height: Image height
        
    Returns:
        (x_center, y_center, width, height) in normalized coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate center coordinates
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    
    # Calculate width and height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height


def create_yolo_label(detections: List[Dict], class_index: int, 
                     img_width: int, img_height: int,
                     target_class: str = 'code_container_h') -> List[str]:
    """
    Create YOLO format label lines from detections.
    
    Args:
        detections: List of detection dictionaries
        class_index: Class index to use for labels
        img_width: Image width
        img_height: Image height
        target_class: Only include detections of this class
        
    Returns:
        List of YOLO format label lines
    """
    label_lines = []
    
    for det in detections:
        # Only process detections of the target class
        if det['class'] == target_class:
            # Convert bbox to YOLO format
            x_center, y_center, width, height = convert_to_yolo_format(
                det['bbox'], img_width, img_height
            )
            
            # Create YOLO label line: class_index x_center y_center width height
            label_line = f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            label_lines.append(label_line)
    
    return label_lines


def relabel_all_images(images_dir: Path, labels_dir: Path, model,
                      class_index: int, confidence_threshold: float,
                      use_full_image: bool, logger: logging.Logger) -> Dict:
    """
    Re-label all images in the images directory.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory to save labels
        model: YOLO model for inference
        class_index: Class index for YOLO labels
        confidence_threshold: Confidence threshold for detections
        use_full_image: If True, create label for full image
        logger: Logger instance
        
    Returns:
        Processing statistics
    """
    # Ensure labels directory exists
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    logger.info(f"Found {len(image_files)} images to re-label")
    
    stats = {
        "total_images": len(image_files),
        "images_processed": 0,
        "labels_created": 0,
        "images_with_detections": 0,
        "images_without_detections": 0,
        "total_detections": 0,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        
        try:
            # Create label file path
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            if use_full_image:
                # Skip full-image labeling - we only want actual detections
                logger.warning(f"--use-full-image mode disabled to prevent incorrect labels. Use inference mode instead.")
                continue
            else:
                # Run inference
                results = model(str(image_path), conf=confidence_threshold, verbose=False)
                
                # Get image dimensions
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue
                    
                img_height, img_width = image.shape[:2]
                
                # Process detections
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id
                        }
                        detections.append(detection)
                
                # Create YOLO labels for target class
                label_lines = create_yolo_label(
                    detections, class_index, img_width, img_height, 
                    target_class='code_container_h'
                )
                
                # Write labels to file ONLY if we have actual detections
                if label_lines:
                    with open(label_path, 'w') as f:
                        for line in label_lines:
                            f.write(line + '\n')
                    stats["images_with_detections"] += 1
                    stats["total_detections"] += len(label_lines)
                    stats["labels_created"] += 1
                    logger.debug(f"Created label with {len(label_lines)} detections: {label_path.name}")
                else:
                    # No detections - skip creating label file entirely
                    stats["images_without_detections"] += 1
                    logger.debug(f"No detections found, skipping label creation: {image_path.name}")
            
            stats["images_processed"] += 1
            
            # Progress update
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                logger.info(f"Progress: {idx}/{len(image_files)} images "
                          f"({rate:.1f} images/sec)")
                
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            continue
    
    stats["processing_time"] = time.time() - start_time
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print processing statistics."""
    logger.info("\n=== RE-LABELING STATISTICS ===")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Images processed: {stats['images_processed']}")
    logger.info(f"Labels created: {stats['labels_created']}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Processing time: {stats['processing_time']:.1f}s")
    
    if stats['images_processed'] > 0:
        rate = stats['images_processed'] / stats['processing_time']
        logger.info(f"Processing rate: {rate:.1f} images/sec")
        
        avg_detections = stats['total_detections'] / stats['images_processed']
        logger.info(f"Average detections per image: {avg_detections:.1f}")
        
        if 'images_with_detections' in stats:
            logger.info(f"Images with detections: {stats['images_with_detections']}")
            logger.info(f"Images without detections (fallback): {stats['images_without_detections']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-label all images in the 01_labelled folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--labelled-dir",
        type=Path,
        default=LABELLED_DIR,
        help="Base directory containing images and labels subdirectories"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO model for inference"
    )
    
    parser.add_argument(
        "--class-index",
        type=int,
        default=CLASS_NAMES_TO_INDEX.get('code_container_h', 0),
        help="Class index for YOLO labels"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.2,
        help="Confidence threshold for detections (higher = stricter, better quality labels)"
    )
    
    parser.add_argument(
        "--use-full-image",
        action="store_true",
        help="Create label for full image instead of running inference"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    images_dir = args.labelled_dir / "images"
    labels_dir = args.labelled_dir / "labels"
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Re-label All Images ===")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels directory: {labels_dir}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Class index: {args.class_index}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Use full image: {args.use_full_image}")
    
    # Validate directories
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1
    
    if not args.model_path.exists() and not args.use_full_image:
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    try:
        # Load model if needed
        model = None
        if not args.use_full_image:
            model = load_yolo_model(args.model_path, logger)
        
        # Clear existing labels directory and recreate
        if labels_dir.exists():
            import shutil
            shutil.rmtree(labels_dir)
            logger.info("Cleared existing labels directory")
        
        # Re-label all images
        logger.info("\nStarting re-labeling process...")
        stats = relabel_all_images(
            images_dir, labels_dir, model,
            args.class_index, args.confidence, args.use_full_image,
            logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        logger.info(f"\n✅ Re-labeling completed!")
        logger.info(f"All labels saved to: {labels_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Re-labeling failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())