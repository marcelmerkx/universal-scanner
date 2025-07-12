#!/usr/bin/env python3
"""
Recover original files for images with special suffixes and create YOLO labels.

This script:
1. Scans data/detection/training_data/00_raw/code_container_h for files with suffixes:
   -checksummissing, -multi, -miss
2. Extracts the original filename by removing the suffix
3. Finds the original file in data/detection/containerdoors
4. Copies original to data/detection/training_data/01_labelled/images
5. Creates YOLO format labels in data/detection/training_data/01_labelled/labels

Examples:
    BSIU2464013-1-checksummissing.jpg → find BSIU2464013-1.jpg in containerdoors
    AXIU1349971-1-multi.jpg → find AXIU1349971-1.jpg in containerdoors
    CRXU9452273-1-miss.jpg → find CRXU9452273-1.jpg in containerdoors

Usage:
    python3 data/detection/scripts/recover_special_suffix_files.py
    python3 data/detection/scripts/recover_special_suffix_files.py --use-full-image
"""

import argparse
import logging
import re
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
SCAN_DIR = Path("data/detection/training_data/00_raw/code_container_h")
OUTPUT_BASE_DIR = Path("data/detection/training_data/01_labelled")
DEFAULT_MODEL_PATH = Path("data/detection/models/detection_320_grayscale_tilted-09-07-2025.pt")
LOG_DIR = Path("data/detection/logs")

# Special suffixes to look for
SPECIAL_SUFFIXES = ['-checksummissing', '-multi', '-miss']

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
    log_file = LOG_DIR / "recover_special_suffix_files.log"
    
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


def find_files_with_special_suffixes(scan_dir: Path, logger: logging.Logger) -> List[Tuple[Path, str, str]]:
    """
    Find files with special suffixes in the scan directory.
    
    Args:
        scan_dir: Directory to scan for files with special suffixes
        logger: Logger instance
        
    Returns:
        List of tuples: (file_path, original_name, suffix_found)
    """
    special_files = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in scan_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    logger.info(f"Scanning {len(image_files)} files in {scan_dir}")
    
    for file_path in image_files:
        # Check if filename contains any special suffix
        filename = file_path.stem  # filename without extension
        
        for suffix in SPECIAL_SUFFIXES:
            if suffix in filename:
                # Extract original filename by removing the suffix
                original_name = filename.replace(suffix, '')
                special_files.append((file_path, original_name, suffix))
                logger.info(f"Found special file: {file_path.name} → original: {original_name}")
                break  # Only match first suffix found
    
    logger.info(f"Found {len(special_files)} files with special suffixes")
    return special_files


def find_original_files(special_files: List[Tuple[Path, str, str]], 
                       source_dir: Path, logger: logging.Logger) -> List[Tuple[Path, Path, str]]:
    """
    Find original files in source directory that match the extracted names.
    
    Args:
        special_files: List of (file_path, original_name, suffix) tuples
        source_dir: Directory containing original images
        logger: Logger instance
        
    Returns:
        List of tuples: (special_file_path, original_file_path, suffix)
    """
    # Build index of source files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    source_files = {}
    for f in source_dir.iterdir():
        if f.is_file() and f.suffix.lower() in image_extensions:
            source_files[f.stem] = f
    
    matched_files = []
    
    for special_file_path, original_name, suffix in special_files:
        if original_name in source_files:
            original_file_path = source_files[original_name]
            matched_files.append((special_file_path, original_file_path, suffix))
            logger.info(f"Matched: {special_file_path.name} → {original_file_path.name}")
        else:
            logger.warning(f"Could not find original file for: {original_name}")
    
    logger.info(f"Successfully matched {len(matched_files)} files")
    return matched_files


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


def process_matched_files(matched_files: List[Tuple[Path, Path, str]], 
                         model, output_base_dir: Path, class_index: int,
                         confidence_threshold: float, use_full_image: bool,
                         logger: logging.Logger) -> Dict:
    """
    Process matched files: copy originals and create labels.
    
    Args:
        matched_files: List of (special_file, original_file, suffix) tuples
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
        "total_files": len(matched_files),
        "files_processed": 0,
        "labels_created": 0,
        "files_with_detections": 0,
        "files_without_detections": 0,
        "suffix_counts": {}
    }
    
    for idx, (special_file_path, original_file_path, suffix) in enumerate(matched_files, 1):
        logger.info(f"Processing {idx}/{len(matched_files)}: {original_file_path.name} (reason: {suffix})")
        
        # Count suffixes
        if suffix not in stats["suffix_counts"]:
            stats["suffix_counts"][suffix] = 0
        stats["suffix_counts"][suffix] += 1
        
        try:
            # Check if file already exists in output
            output_image_path = images_dir / original_file_path.name
            label_path = labels_dir / f"{original_file_path.stem}.txt"
            
            if output_image_path.exists():
                logger.info(f"File already exists in output: {original_file_path.name}")
                # Still create label if it doesn't exist
                if not label_path.exists():
                    logger.info(f"Creating missing label for existing image: {original_file_path.name}")
                else:
                    logger.info(f"Label also exists, skipping: {original_file_path.name}")
                    stats["files_processed"] += 1
                    continue
            else:
                # Copy original image to output directory
                shutil.copy2(original_file_path, output_image_path)
                logger.info(f"Copied image: {original_file_path.name}")
            
            if use_full_image:
                # Create label for full image
                with open(label_path, 'w') as f:
                    # YOLO format: class x_center y_center width height
                    # For full image: center at 0.5, 0.5 with width and height of 1.0
                    f.write(f"{class_index} 0.5 0.5 1.0 1.0\n")
                stats["labels_created"] += 1
                logger.info(f"Created full-image label: {label_path.name}")
            else:
                # Run inference to get detections
                results = model(str(original_file_path), conf=confidence_threshold, verbose=False)
                
                # Get image dimensions
                image = cv2.imread(str(original_file_path))
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
                    stats["files_with_detections"] += 1
                    logger.info(f"Created detection label with {len(label_lines)} boxes: {label_path.name}")
                else:
                    # No detections of target class
                    logger.warning(f"No code_container_h detections found in {original_file_path.name}")
                    stats["files_without_detections"] += 1
                    # Create empty label file or full image label as fallback
                    with open(label_path, 'w') as f:
                        f.write(f"{class_index} 0.5 0.5 1.0 1.0\n")
                    stats["labels_created"] += 1
                    logger.info(f"Created fallback full-image label: {label_path.name}")
            
            stats["files_processed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to process {original_file_path.name}: {e}")
            continue
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Recover original files for images with special suffixes and create YOLO labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=SCAN_DIR,
        help="Directory to scan for files with special suffixes"
    )
    
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Source directory containing original images"
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
    
    logger.info("=== Recover Special Suffix Files ===")
    logger.info(f"Scan directory: {args.scan_dir}")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Special suffixes: {SPECIAL_SUFFIXES}")
    logger.info(f"Class index: {args.class_index}")
    logger.info(f"Use full image: {args.use_full_image}")
    
    # Validate directories
    if not args.scan_dir.exists():
        logger.error(f"Scan directory not found: {args.scan_dir}")
        return 1
    
    if not args.source_dir.exists():
        logger.error(f"Source directory not found: {args.source_dir}")
        return 1
    
    try:
        # Find files with special suffixes
        special_files = find_files_with_special_suffixes(args.scan_dir, logger)
        
        if not special_files:
            logger.info("No files with special suffixes found. Nothing to process.")
            return 0
        
        # Find corresponding original files
        matched_files = find_original_files(special_files, args.source_dir, logger)
        
        if not matched_files:
            logger.info("No matching original files found. Nothing to process.")
            return 0
        
        # Load model if needed
        model = None
        if not args.use_full_image:
            model = load_yolo_model(args.model_path, logger)
        
        # Process matched files
        logger.info("\nProcessing matched files...")
        stats = process_matched_files(
            matched_files, model, args.output_dir, 
            args.class_index, args.confidence, args.use_full_image,
            logger
        )
        
        # Print statistics
        logger.info("\n=== PROCESSING STATISTICS ===")
        logger.info(f"Total matched files: {stats['total_files']}")
        logger.info(f"Files processed: {stats['files_processed']}")
        logger.info(f"Labels created: {stats['labels_created']}")
        if not args.use_full_image:
            logger.info(f"Files with detections: {stats['files_with_detections']}")
            logger.info(f"Files without detections: {stats['files_without_detections']}")
        
        logger.info("\nFiles by suffix:")
        for suffix, count in stats["suffix_counts"].items():
            logger.info(f"  {suffix}: {count}")
        
        logger.info(f"\n✅ Processing completed!")
        logger.info(f"Images saved to: {args.output_dir / 'images'}")
        logger.info(f"Labels saved to: {args.output_dir / 'labels'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())