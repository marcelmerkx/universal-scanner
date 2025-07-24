#!/usr/bin/env python3
"""
Download annotations from one Roboflow project, adjust bounding box widths, and re-upload to another project.

This script:
1. Downloads images and annotations from the "vertical-ocr" project
2. Reduces the width of character bounding boxes by 10% (symmetrically)
3. Re-uploads to the "container-ocr-bt9wl" project

Usage:
    python3 data/detection/scripts/adjust_and_transfer_annotations.py --limit 10
    python3 data/detection/scripts/adjust_and_transfer_annotations.py --all
"""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed. Run: pip install roboflow")
    exit(1)

# Configuration
LOG_DIR = Path("data/detection/logs")
TEMP_DOWNLOAD_DIR = Path("data/detection/temp_downloads")

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / '.env')

# Define labelmap for OCR classes (0-9 digits, 10-35 letters A-Z)
LABELMAP = {i: name for i, name in enumerate([
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
])}


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "adjust_and_transfer_annotations.log"
    
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


def download_project_data(api_key: str, workspace: str, source_project: str, 
                         version: str, download_dir: Path, logger: logging.Logger) -> Path:
    """
    Download project data from Roboflow.
    
    Returns:
        Path to the downloaded dataset directory
    """
    logger.info(f"Downloading from project: {workspace}/{source_project} version {version}")
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(source_project)
    dataset = project.version(version)
    
    # Download dataset in YOLO format
    # The download method returns the path as a string when successful
    dataset.download("yolov8", location=str(download_dir), overwrite=True)
    
    # The downloaded data should be in a subdirectory with the project name
    # Look for the actual dataset directory
    possible_paths = [
        download_dir / f"{source_project}-{version}",
        download_dir / source_project,
        download_dir
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "data.yaml").exists():
            logger.info(f"Found dataset at: {path}")
            return path
    
    # If we can't find data.yaml, look for any directory with train/valid/test structure
    for subdir in download_dir.iterdir():
        if subdir.is_dir() and any((subdir / split).exists() for split in ["train", "valid", "test"]):
            logger.info(f"Found dataset at: {subdir}")
            return subdir
    
    logger.error(f"Could not find downloaded dataset in {download_dir}")
    raise ValueError(f"Dataset structure not found in {download_dir}")


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse YOLO label file and return list of detections.
    
    Returns:
        List of (class_id, center_x, center_y, width, height) tuples
    """
    detections = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                detections.append((class_id, center_x, center_y, width, height))
    
    return detections


def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1, box2: (center_x, center_y, width, height) in normalized coordinates
        
    Returns:
        IoU value between 0 and 1
    """
    # Convert from center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def remove_duplicate_labels(detections: List[Tuple[int, float, float, float, float]], 
                          iou_threshold: float = 0.05,  # Much lower threshold to catch any overlap
                          max_labels: int = 11) -> List[Tuple[int, float, float, float, float]]:
    """
    Remove duplicate labels based on IoU threshold and limit to max_labels.
    For vertical containers, sorts by Y-coordinate (top to bottom).
    
    Args:
        detections: List of (class_id, center_x, center_y, width, height) tuples
        iou_threshold: IoU threshold for considering boxes as duplicates (0.05 = 5% overlap)
        max_labels: Maximum number of labels to keep
        
    Returns:
        List of filtered detections
    """
    if len(detections) == 0:
        return detections
    
    # Sort by area (larger boxes first) to prioritize keeping larger detections
    detections_with_area = []
    for det in detections:
        area = det[3] * det[4]  # width * height
        detections_with_area.append((det, area))
    
    detections_with_area.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates based on IoU - ANY overlap is removed
    filtered = []
    
    for det, area in detections_with_area:
        # Check if this detection overlaps with any already kept detection
        is_duplicate = False
        
        for kept_det in filtered:
            # Calculate IoU between current detection and kept detection
            box1 = (det[1], det[2], det[3], det[4])  # center_x, center_y, width, height
            box2 = (kept_det[1], kept_det[2], kept_det[3], kept_det[4])
            
            iou = calculate_iou(box1, box2)
            
            if iou > iou_threshold:  # Any overlap above 5%
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(det)
            
            # Stop if we've reached max_labels
            if len(filtered) >= max_labels:
                break
    
    # Sort by y-coordinate to maintain top-to-bottom order for vertical containers
    filtered.sort(key=lambda x: x[2])  # Sort by center_y
    
    return filtered


def adjust_bbox_width(detections: List[Tuple[int, float, float, float, float]], 
                     width_reduction: float = 0.1) -> List[Tuple[int, float, float, float, float]]:
    """
    Reduce the width of bounding boxes by the specified percentage.
    The reduction is applied symmetrically to keep boxes centered.
    
    Args:
        detections: List of (class_id, center_x, center_y, width, height) tuples
        width_reduction: Percentage to reduce width (0.10 = 10%)
        
    Returns:
        List of adjusted detections
    """
    adjusted = []
    
    for class_id, center_x, center_y, width, height in detections:
        # Reduce width by the specified percentage
        new_width = width * (1.0 - width_reduction)
        
        # Keep the center position the same
        adjusted.append((class_id, center_x, center_y, new_width, height))
    
    return adjusted


def write_yolo_label(label_path: Path, detections: List[Tuple[int, float, float, float, float]]):
    """Write detections to YOLO format label file."""
    with open(label_path, 'w') as f:
        for class_id, center_x, center_y, width, height in detections:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def process_annotations(dataset_dir: Path, output_dir: Path, width_reduction: float, 
                       overlap_threshold: float, limit: Optional[int], logger: logging.Logger) -> Dict:
    """
    Process annotations to reduce bounding box widths and remove duplicates.
    
    Returns:
        Processing statistics
    """
    # Find split directories (train, valid, test)
    splits = ['train', 'valid', 'test']
    
    stats = {
        "total_images": 0,
        "processed_images": 0,
        "total_boxes": 0,
        "adjusted_boxes": 0,
        "removed_duplicates": 0,
        "images_with_duplicates": 0,
        "failed": 0
    }
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.debug(f"Split '{split}' not found, skipping")
            continue
            
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Missing images or labels directory in {split}, skipping")
            continue
        
        # Create output directories for this split
        output_images_dir = output_dir / split / "images"
        output_labels_dir = output_dir / split / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        
        if limit and stats["processed_images"] >= limit:
            logger.info(f"Reached limit of {limit} images")
            break
        
        logger.info(f"\nProcessing {split} split: {len(image_files)} images")
        
        for image_path in image_files:
            if limit and stats["processed_images"] >= limit:
                break
                
            stats["total_images"] += 1
            
            # Find corresponding label
            label_name = image_path.stem + ".txt"
            label_path = labels_dir / label_name
            
            if not label_path.exists():
                logger.warning(f"No label found for {image_path.name}")
                continue
            
            try:
                # Parse original annotations
                detections = parse_yolo_label(label_path)
                original_count = len(detections)
                stats["total_boxes"] += original_count
                
                # Always remove duplicate/overlapping labels
                filtered_detections = remove_duplicate_labels(detections, overlap_threshold)
                removed_count = original_count - len(filtered_detections)
                if removed_count > 0:
                    stats["removed_duplicates"] += removed_count
                    stats["images_with_duplicates"] += 1
                    logger.info(f"{image_path.name}: Removed {removed_count} overlapping labels (had {original_count}, kept {len(filtered_detections)})")
                
                # Adjust bounding box widths
                adjusted_detections = adjust_bbox_width(filtered_detections, width_reduction)
                stats["adjusted_boxes"] += len(adjusted_detections)
                
                # Rename image to proper format (first 11 chars + .jpg)
                original_stem = image_path.stem
                # Extract first 11 characters (container code)
                if len(original_stem) >= 11:
                    new_name = original_stem[:11] + ".jpg"
                else:
                    new_name = original_stem + ".jpg"
                
                # Log renaming if name changed
                if new_name != image_path.name:
                    logger.debug(f"Renaming: {image_path.name} → {new_name}")
                
                # Copy image with new name
                output_image_path = output_images_dir / new_name
                shutil.copy2(image_path, output_image_path)
                
                # Write adjusted labels with matching name
                output_label_name = new_name.replace(".jpg", ".txt")
                output_label_path = output_labels_dir / output_label_name
                write_yolo_label(output_label_path, adjusted_detections)
                
                stats["processed_images"] += 1
                
                if stats["processed_images"] % 10 == 0:
                    logger.info(f"Processed {stats['processed_images']} images...")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                stats["failed"] += 1
    
    return stats


def upload_to_target_project(api_key: str, workspace: str, target_project: str,
                           processed_dir: Path, split_name: str, logger: logging.Logger) -> Dict:
    """
    Upload processed images and annotations to target project.
    
    Returns:
        Upload statistics
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(target_project)
    
    stats = {
        "uploaded": 0,
        "failed": 0
    }
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
            
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists():
            continue
        
        # Get list of images
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        
        logger.info(f"\nUploading {len(image_files)} images from {split} split to {split_name}")
        
        for image_path in image_files:
            # Find corresponding label
            label_path = labels_dir / (image_path.stem + ".txt")
            
            if not label_path.exists():
                logger.warning(f"No label found for {image_path.name}, skipping")
                continue
            
            try:
                # Upload with labelmap
                result = project.single_upload(
                    image_path=str(image_path),
                    annotation_path=str(label_path),
                    annotation_labelmap=LABELMAP,
                    split=split_name,
                    num_retry_uploads=1
                )
                
                stats["uploaded"] += 1
                logger.debug(f"Uploaded: {image_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to upload {image_path.name}: {e}")
                stats["failed"] += 1
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adjust bounding box widths and transfer between Roboflow projects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--api-key",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
        default=os.environ.get("ROBOFLOW_API_KEY")
    )
    
    parser.add_argument(
        "--workspace",
        default="cargosnap",
        help="Roboflow workspace name"
    )
    
    parser.add_argument(
        "--source-project",
        default="vertical-ocr",
        help="Source Roboflow project ID"
    )
    
    parser.add_argument(
        "--source-version",
        type=int,
        default=1,
        help="Source project version number"
    )
    
    parser.add_argument(
        "--target-project",
        default="container-ocr-bt9wl",
        help="Target Roboflow project ID"
    )
    
    parser.add_argument(
        "--width-reduction",
        type=float,
        default=0.15,
        help="Percentage to reduce bbox width (0.15 = 15%%)"
    )
    
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.25,
        help="IoU threshold for removing overlapping boxes (0.05 = 5%% overlap)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of images to process (for testing)"
    )
    
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
        help="Target dataset split for upload"
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if data already exists in temp directory"
    )
    
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload step (only process annotations)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all images (no limit)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Adjust and Transfer Annotations ===")
    logger.info(f"Source: {args.workspace}/{args.source_project} v{args.source_version}")
    logger.info(f"Target: {args.workspace}/{args.target_project}")
    logger.info(f"Width reduction: {args.width_reduction * 100:.0f}%")
    logger.info(f"Process limit: {args.limit if args.limit else 'All images' if args.all else '10 (default)'}")
    
    # Set default limit if not specified
    if not args.all and not args.limit:
        args.limit = 10
        logger.info("Using default limit of 10 images for testing")
    
    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: Roboflow API key is required.")
        logger.error("Provide it via --api-key or set ROBOFLOW_API_KEY environment variable.")
        return 1
    
    try:
        # Step 1: Download from source project
        download_dir = TEMP_DOWNLOAD_DIR / f"{args.source_project}_v{args.source_version}"
        
        if args.skip_download and download_dir.exists():
            logger.info(f"Skipping download, using existing data in {download_dir}")
            dataset_dir = download_dir
        else:
            # Clean up any existing download directory
            if download_dir.exists():
                shutil.rmtree(download_dir)
            download_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_dir = download_project_data(
                args.api_key, args.workspace, args.source_project,
                args.source_version, download_dir, logger
            )
        
        # Step 2: Process annotations
        logger.info("\n=== Processing Annotations ===")
        processed_dir = TEMP_DOWNLOAD_DIR / f"{args.source_project}_processed"
        
        # Clean up any existing processed directory
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        
        process_stats = process_annotations(
            dataset_dir, processed_dir, args.width_reduction, args.overlap_threshold, args.limit, logger
        )
        
        logger.info("\n=== Processing Statistics ===")
        logger.info(f"Total images found: {process_stats['total_images']}")
        logger.info(f"Images processed: {process_stats['processed_images']}")
        logger.info(f"Total boxes: {process_stats['total_boxes']}")
        logger.info(f"Duplicate boxes removed: {process_stats['removed_duplicates']}")
        logger.info(f"Images with duplicates: {process_stats['images_with_duplicates']}")
        logger.info(f"Final boxes (after filtering): {process_stats['adjusted_boxes']}")
        logger.info(f"Failed: {process_stats['failed']}")
        
        if process_stats['processed_images'] > 0:
            avg_boxes_before = process_stats['total_boxes'] / process_stats['processed_images']
            avg_boxes_after = process_stats['adjusted_boxes'] / process_stats['processed_images']
            logger.info(f"Average boxes per image: {avg_boxes_before:.1f} → {avg_boxes_after:.1f}")
        
        # Step 3: Upload to target project
        if not args.skip_upload and process_stats['processed_images'] > 0:
            logger.info(f"\n=== Uploading to Target Project ===")
            logger.info(f"Target: {args.workspace}/{args.target_project}")
            logger.info(f"Split: {args.split}")
            
            upload_stats = upload_to_target_project(
                args.api_key, args.workspace, args.target_project,
                processed_dir, args.split, logger
            )
            
            logger.info("\n=== Upload Statistics ===")
            logger.info(f"Successfully uploaded: {upload_stats['uploaded']}")
            logger.info(f"Failed uploads: {upload_stats['failed']}")
            
            if upload_stats['uploaded'] > 0:
                logger.info(f"\n✅ Successfully transferred {upload_stats['uploaded']} images!")
                logger.info(f"Check your target project: https://app.roboflow.com/{args.workspace}/{args.target_project}")
        else:
            if args.skip_upload:
                logger.info("\nSkipping upload as requested")
            else:
                logger.warning("\nNo images to upload")
        
        # Offer to clean up
        logger.info(f"\nProcessed data saved in: {processed_dir}")
        logger.info("You can inspect the adjusted annotations before uploading")
        
        return 0
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())