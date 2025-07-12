#!/usr/bin/env python3
"""
Recover deleted images and create YOLO labels for them.

This script:
1. Compares source folder (containerdoors) with target folder (00_raw/code_container_h)
2. Finds images that were deleted from target
3. Runs inference to get bounding boxes
4. Copies missing images to 01_labelled/images
5. Creates YOLO format label files in 01_labelled/labels

Usage:
    python3 data/detection/scripts/recover_deleted_with_labels.py
    python3 data/detection/scripts/recover_deleted_with_labels.py --class-index 1
    python3 data/detection/scripts/recover_deleted_with_labels.py --use-full-image
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed. Run: pip install ultralytics")
    exit(1)

# Configuration
SOURCE_DIR = Path("data/detection/containerdoors")
TARGET_DIR = Path("data/detection/training_data/00_raw/code_container_h")
OUTPUT_BASE_DIR = Path("data/detection/training_data/01_labelled")
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
    log_file = LOG_DIR / "recover_deleted_with_labels.log"
    
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


def find_missing_images(source_dir: Path, target_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    Find images that exist in source but not in target.
    
    Args:
        source_dir: Directory containing original images
        target_dir: Directory where some images were deleted
        logger: Logger instance
        
    Returns:
        List of paths to missing images
    """
    # Get all image files from source
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    source_images = {f.stem: f for f in source_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions}
    
    # Get all image files from target
    target_images = {f.stem: f for f in target_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions}
    
    # Find missing images (handle case where stem might have suffix like _0, _1)
    missing_images = []
    for stem, source_path in source_images.items():
        # Check if this stem exists in target (exact match)
        if stem not in target_images:
            # Also check if any file starts with this stem (for _0, _1 suffixes)
            stem_found = False
            for target_stem in target_images:
                if target_stem.startswith(stem):
                    stem_found = True
                    break
            
            if not stem_found:
                missing_images.append(source_path)
    
    logger.info(f"Found {len(source_images)} images in source directory")
    logger.info(f"Found {len(target_images)} images in target directory")
    logger.info(f"Found {len(missing_images)} missing images")
    
    return missing_images


def load_yolo_model(model_path: Path, logger: logging.Logger):
    """Load YOLO model for inference."""
    try:
        logger.info(f"Loading YOLO model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(str(model_path))
        logger.info("✅ Model loaded successfully")
        
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


def process_missing_images(missing_images: List[Path], model, 
                          output_base_dir: Path, class_index: int,
                          confidence_threshold: float, use_full_image: bool,
                          logger: logging.Logger) -> Dict:
    """
    Process missing images: copy them and create labels.
    
    Args:
        missing_images: List of missing image paths
        model: YOLO model for inference
        output_base_dir: Base directory for output
        class_index: Class index for YOLO labels
        confidence_threshold: Confidence threshold for detections
        use_full_image: If True, create label for full image instead of detections
        logger: Logger instance
        
    Returns:
        Processing statistics
    """
    # Create output directories
    images_dir = output_base_dir / "images"
    labels_dir = output_base_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_images": len(missing_images),
        "images_processed": 0,
        "labels_created": 0,
        "images_with_detections": 0,
        "images_without_detections": 0
    }
    
    for idx, image_path in enumerate(missing_images, 1):
        logger.info(f"Processing {idx}/{len(missing_images)}: {image_path.name}")
        
        try:
            # Copy image to output directory
            output_image_path = images_dir / image_path.name
            shutil.copy2(image_path, output_image_path)
            
            # Create label file path
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            if use_full_image:
                # Create label for full image
                with open(label_path, 'w') as f:
                    # YOLO format: class x_center y_center width height
                    # For full image: center at 0.5, 0.5 with width and height of 1.0
                    f.write(f"{class_index} 0.5 0.5 1.0 1.0\n")
                stats["labels_created"] += 1
            else:
                # Run inference to get detections
                results = model(str(image_path), conf=confidence_threshold, verbose=False)
                
                # Get image dimensions
                image = cv2.imread(str(image_path))
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
                
                # Create YOLO labels
                label_lines = create_yolo_label(
                    detections, class_index, img_width, img_height, 
                    target_class='code_container_h'
                )
                
                if label_lines:
                    # Write labels to file
                    with open(label_path, 'w') as f:
                        for line in label_lines:
                            f.write(line + '\n')
                    stats["labels_created"] += 1
                    stats["images_with_detections"] += 1
                else:
                    # No detections of target class - create empty label file
                    # or skip based on your preference
                    logger.warning(f"No code_container_h detections found in {image_path.name}")
                    stats["images_without_detections"] += 1
                    # Optionally create empty label file
                    # open(label_path, 'w').close()
            
            stats["images_processed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            continue
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Recover deleted images and create YOLO labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Source directory containing original images"
    )
    
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=TARGET_DIR,
        help="Target directory where images were deleted"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE_DIR,
        help="Output directory for images and labels"
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
        default=0.25,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--use-full-image",
        action="store_true",
        help="Create label for full image instead of running inference"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Recover Deleted Images with Labels ===")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Target directory: {args.target_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Class index: {args.class_index}")
    logger.info(f"Use full image: {args.use_full_image}")
    
    # Validate directories
    if not args.source_dir.exists():
        logger.error(f"Source directory not found: {args.source_dir}")
        return 1
    
    if not args.target_dir.exists():
        logger.error(f"Target directory not found: {args.target_dir}")
        return 1
    
    try:
        # Find missing images
        missing_images = find_missing_images(args.source_dir, args.target_dir, logger)
        
        if not missing_images:
            logger.info("No missing images found. Nothing to process.")
            return 0
        
        # Load model if needed
        model = None
        if not args.use_full_image:
            model = load_yolo_model(args.model_path, logger)
        
        # Process missing images
        logger.info("\nProcessing missing images...")
        stats = process_missing_images(
            missing_images, model, args.output_dir, 
            args.class_index, args.confidence, args.use_full_image,
            logger
        )
        
        # Print statistics
        logger.info("\n=== PROCESSING STATISTICS ===")
        logger.info(f"Total missing images: {stats['total_images']}")
        logger.info(f"Images processed: {stats['images_processed']}")
        logger.info(f"Labels created: {stats['labels_created']}")
        if not args.use_full_image:
            logger.info(f"Images with detections: {stats['images_with_detections']}")
            logger.info(f"Images without detections: {stats['images_without_detections']}")
        
        logger.info(f"\n✅ Processing completed!")
        logger.info(f"Images saved to: {args.output_dir / 'images'}")
        logger.info(f"Labels saved to: {args.output_dir / 'labels'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())