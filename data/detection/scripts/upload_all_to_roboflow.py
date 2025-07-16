#!/usr/bin/env python3
"""
Upload all images (labelled and unlabelled) to Roboflow project.

This script uploads all images from the 01_labelled/images folder to Roboflow:
- Images with corresponding .txt labels are uploaded with annotations
- Images without labels are uploaded without annotations for manual annotation

Usage:
    python3 data/detection/scripts/upload_all_to_roboflow.py
    python3 data/detection/scripts/upload_all_to_roboflow.py --split valid
    python3 data/detection/scripts/upload_all_to_roboflow.py --batch-size 10
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed. Run: pip install roboflow")
    exit(1)

# Configuration
LABELLED_DIR = Path("data/detection/training_data/01_labelled")
LOG_DIR = Path("data/detection/logs")

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / '.env')


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "upload_all_to_roboflow.log"
    
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


def categorize_images(images_dir: Path, labels_dir: Path, logger: logging.Logger) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """
    Categorize images into labelled and unlabelled.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        logger: Logger instance
        
    Returns:
        Tuple of (labelled_pairs, unlabelled_images)
        - labelled_pairs: List of (image_path, label_path) tuples
        - unlabelled_images: List of image paths without labels
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = {f.stem: f for f in images_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions}
    
    # Find all label files
    label_files = {}
    if labels_dir.exists():
        label_files = {f.stem: f for f in labels_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() == '.txt'}
    
    # Categorize images
    labelled_pairs = []
    unlabelled_images = []
    
    for stem, image_path in image_files.items():
        if stem in label_files:
            labelled_pairs.append((image_path, label_files[stem]))
        else:
            unlabelled_images.append(image_path)
    
    logger.info(f"Found {len(image_files)} total images")
    logger.info(f"Found {len(label_files)} label files")
    logger.info(f"Categorized: {len(labelled_pairs)} labelled, {len(unlabelled_images)} unlabelled")
    
    return labelled_pairs, unlabelled_images


def upload_all_images(api_key: str, workspace: str, project_id: str,
                     labelled_pairs: List[Tuple[Path, Path]], 
                     unlabelled_images: List[Path],
                     batch_size: int, split: str, logger: logging.Logger) -> Dict:
    """
    Upload both labelled and unlabelled images to Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Roboflow project ID
        labelled_pairs: List of (image_path, label_path) tuples
        unlabelled_images: List of image paths without labels
        batch_size: Number of images to upload at once
        split: Dataset split ('train', 'valid', or 'test')
        logger: Logger instance
        
    Returns:
        Upload statistics
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_id)
    
    total_images = len(labelled_pairs) + len(unlabelled_images)
    
    stats = {
        "total_images": total_images,
        "labelled_pairs": len(labelled_pairs),
        "unlabelled_images": len(unlabelled_images),
        "labelled_uploaded": 0,
        "unlabelled_uploaded": 0,
        "labelled_failures": 0,
        "unlabelled_failures": 0,
        "processing_time": 0.0,
        "failed_labelled": [],
        "failed_unlabelled": []
    }
    
    start_time = time.time()
    
    # Phase 1: Upload labelled images with annotations
    if labelled_pairs:
        logger.info(f"\n=== PHASE 1: Uploading {len(labelled_pairs)} labelled images ===")
        logger.info(f"Using single_upload with image_path + annotation_path")
        
        for batch_start in range(0, len(labelled_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(labelled_pairs))
            batch = labelled_pairs[batch_start:batch_end]
            
            logger.info(f"Labelled batch {batch_start//batch_size + 1}: "
                       f"pairs {batch_start + 1}-{batch_end} of {len(labelled_pairs)}")
            
            for idx, (image_path, label_path) in enumerate(batch, batch_start + 1):
                logger.info(f"Uploading labelled {idx}/{len(labelled_pairs)}: {image_path.name} + {label_path.name}")
                
                try:
                    # Upload image with corresponding YOLO label
                    result = project.single_upload(
                        image_path=str(image_path),
                        annotation_path=str(label_path),
                        split=split,
                        num_retry_uploads=1
                    )
                    
                    stats["labelled_uploaded"] += 1
                    logger.debug(f"Successfully uploaded labelled: {image_path.name}")
                    logger.debug(f"Upload result: {result}")
                    
                    # Brief pause to avoid rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Failed to upload labelled {image_path.name}: {e}")
                    stats["labelled_failures"] += 1
                    stats["failed_labelled"].append(str(image_path))
                    continue
            
            # Pause between batches
            if batch_end < len(labelled_pairs):
                logger.info(f"Labelled batch completed. Pausing 2 seconds before next batch...")
                time.sleep(2)
        
        logger.info(f"Phase 1 complete: {stats['labelled_uploaded']} labelled uploaded, {stats['labelled_failures']} failed")
    else:
        logger.info("No labelled images found, skipping Phase 1")
    
    # Brief pause between phases
    if labelled_pairs and unlabelled_images:
        logger.info("\nPausing 3 seconds between phases...")
        time.sleep(3)
    
    # Phase 2: Upload unlabelled images without annotations
    if unlabelled_images:
        logger.info(f"\n=== PHASE 2: Uploading {len(unlabelled_images)} unlabelled images ===")
        logger.info(f"Using single_upload without annotations")
        
        for batch_start in range(0, len(unlabelled_images), batch_size):
            batch_end = min(batch_start + batch_size, len(unlabelled_images))
            batch = unlabelled_images[batch_start:batch_end]
            
            logger.info(f"Unlabelled batch {batch_start//batch_size + 1}: "
                       f"images {batch_start + 1}-{batch_end} of {len(unlabelled_images)}")
            
            for idx, image_path in enumerate(batch, batch_start + 1):
                logger.info(f"Uploading unlabelled {idx}/{len(unlabelled_images)}: {image_path.name}")
                
                try:
                    # Upload image without annotations
                    result = project.single_upload(
                        image_path=str(image_path),
                        split=split,
                        num_retry_uploads=1
                    )
                    
                    stats["unlabelled_uploaded"] += 1
                    logger.debug(f"Successfully uploaded unlabelled: {image_path.name}")
                    logger.debug(f"Upload result: {result}")
                    
                    # Brief pause to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to upload unlabelled {image_path.name}: {e}")
                    stats["unlabelled_failures"] += 1
                    stats["failed_unlabelled"].append(str(image_path))
                    continue
            
            # Pause between batches
            if batch_end < len(unlabelled_images):
                logger.info(f"Unlabelled batch completed. Pausing 2 seconds before next batch...")
                time.sleep(2)
        
        logger.info(f"Phase 2 complete: {stats['unlabelled_uploaded']} unlabelled uploaded, {stats['unlabelled_failures']} failed")
    else:
        logger.info("No unlabelled images found, skipping Phase 2")
    
    stats["processing_time"] = time.time() - start_time
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print upload statistics."""
    logger.info("\n=== UPLOAD STATISTICS ===")
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"  - Labelled pairs: {stats['labelled_pairs']}")
    logger.info(f"  - Unlabelled images: {stats['unlabelled_images']}")
    
    logger.info(f"\nSuccessful uploads:")
    logger.info(f"  - Labelled: {stats['labelled_uploaded']}")
    logger.info(f"  - Unlabelled: {stats['unlabelled_uploaded']}")
    logger.info(f"  - Total: {stats['labelled_uploaded'] + stats['unlabelled_uploaded']}")
    
    logger.info(f"\nFailed uploads:")
    logger.info(f"  - Labelled: {stats['labelled_failures']}")
    logger.info(f"  - Unlabelled: {stats['unlabelled_failures']}")
    logger.info(f"  - Total: {stats['labelled_failures'] + stats['unlabelled_failures']}")
    
    logger.info(f"\nProcessing time: {stats['processing_time']:.1f}s")
    
    if stats['labelled_uploaded'] + stats['unlabelled_uploaded'] > 0:
        rate = (stats['labelled_uploaded'] + stats['unlabelled_uploaded']) / stats['processing_time']
        logger.info(f"Upload rate: {rate:.1f} images/sec")
        
        success_rate = (stats['labelled_uploaded'] + stats['unlabelled_uploaded']) / stats['total_images'] * 100
        logger.info(f"Overall success rate: {success_rate:.1f}%")
    
    # Show failed files
    if stats['failed_labelled']:
        logger.info(f"\nFailed labelled uploads:")
        for failed_file in stats['failed_labelled'][:5]:
            logger.info(f"  - {Path(failed_file).name}")
        if len(stats['failed_labelled']) > 5:
            logger.info(f"  ... and {len(stats['failed_labelled']) - 5} more")
    
    if stats['failed_unlabelled']:
        logger.info(f"\nFailed unlabelled uploads:")
        for failed_file in stats['failed_unlabelled'][:5]:
            logger.info(f"  - {Path(failed_file).name}")
        if len(stats['failed_unlabelled']) > 5:
            logger.info(f"  ... and {len(stats['failed_unlabelled']) - 5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload all images (labelled and unlabelled) to Roboflow project",
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
        "--project",
        default="unified-detection-0zmvz",
        help="Roboflow project ID"
    )
    
    parser.add_argument(
        "--labelled-dir",
        type=Path,
        default=LABELLED_DIR,
        help="Directory containing images and labels subdirectories"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of images to upload in each batch"
    )
    
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
        help="Dataset split for uploaded images"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but don't actually upload to Roboflow"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    images_dir = args.labelled_dir / "images"
    labels_dir = args.labelled_dir / "labels"
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Upload All Images to Roboflow ===")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels directory: {labels_dir}")
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Project: {args.project}")
    logger.info(f"Dataset split: {args.split}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Check if API key is provided
    if not args.api_key and not args.dry_run:
        logger.error("Error: Roboflow API key is required.")
        logger.error("Provide it via --api-key or set ROBOFLOW_API_KEY environment variable.")
        return 1
    
    # Validate directories
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1
    
    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        logger.warning("Will treat all images as unlabelled")
    
    try:
        # Categorize images
        labelled_pairs, unlabelled_images = categorize_images(images_dir, labels_dir, logger)
        
        if not labelled_pairs and not unlabelled_images:
            logger.error("No images found to upload.")
            return 1
        
        if args.dry_run:
            logger.info(f"\n✅ Dry run completed!")
            logger.info(f"Would upload {len(labelled_pairs)} labelled pairs and {len(unlabelled_images)} unlabelled images")
            logger.info(f"All uploads would go to '{args.split}' split")
            
            # Show examples
            if labelled_pairs:
                logger.info("\nExample labelled pairs:")
                for i, (img_path, lbl_path) in enumerate(labelled_pairs[:3]):
                    logger.info(f"  {img_path.name} + {lbl_path.name}")
                if len(labelled_pairs) > 3:
                    logger.info(f"  ... and {len(labelled_pairs) - 3} more labelled pairs")
            
            if unlabelled_images:
                logger.info("\nExample unlabelled images:")
                for i, img_path in enumerate(unlabelled_images[:3]):
                    logger.info(f"  {img_path.name}")
                if len(unlabelled_images) > 3:
                    logger.info(f"  ... and {len(unlabelled_images) - 3} more unlabelled images")
            
            return 0
        
        # Upload to Roboflow
        logger.info("\nStarting upload to Roboflow...")
        
        stats = upload_all_images(
            args.api_key, args.workspace, args.project,
            labelled_pairs, unlabelled_images, args.batch_size, args.split, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        total_failures = stats["labelled_failures"] + stats["unlabelled_failures"]
        if total_failures == 0:
            logger.info(f"\n✅ All images uploaded successfully!")
        else:
            logger.warning(f"\n⚠️  Upload completed with {total_failures} total failures")
        
        logger.info(f"Check your Roboflow project: https://app.roboflow.com/{args.workspace}/{args.project}")
        logger.info("Note: Unlabelled images can be manually annotated in the Roboflow interface")
        
        return 0 if total_failures == 0 else 1
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())