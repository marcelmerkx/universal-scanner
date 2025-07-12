#!/usr/bin/env python3
"""
Upload unlabelled images to Roboflow project.

This script finds images in the 01_labelled/images folder that don't have
corresponding label files and uploads them to Roboflow without annotations.

Usage:
    python3 data/detection/scripts/upload_unlabelled_images.py
    python3 data/detection/scripts/upload_unlabelled_images.py --split valid
    python3 data/detection/scripts/upload_unlabelled_images.py --batch-size 20
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
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
    log_file = LOG_DIR / "upload_unlabelled_images.log"
    
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


def find_unlabelled_images(images_dir: Path, labels_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    Find images that don't have corresponding label files.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        logger: Logger instance
        
    Returns:
        List of image paths without labels
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = {f.stem: f for f in images_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions}
    
    # Find all label files
    label_files = {f.stem: f for f in labels_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() == '.txt'}
    
    # Find images without labels
    unlabelled_images = []
    for stem, image_path in image_files.items():
        if stem not in label_files:
            unlabelled_images.append(image_path)
    
    logger.info(f"Found {len(image_files)} total images")
    logger.info(f"Found {len(label_files)} label files")
    logger.info(f"Found {len(unlabelled_images)} unlabelled images")
    
    return unlabelled_images


def upload_unlabelled_images(api_key: str, workspace: str, project_id: str,
                            unlabelled_images: List[Path], batch_size: int, 
                            split: str, logger: logging.Logger) -> Dict:
    """
    Upload unlabelled images to Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Roboflow project ID
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
    
    stats = {
        "total_images": len(unlabelled_images),
        "uploaded_successfully": 0,
        "upload_failures": 0,
        "processing_time": 0.0,
        "failed_files": []
    }
    
    start_time = time.time()
    
    logger.info(f"=== Uploading {len(unlabelled_images)} unlabelled images ===")
    logger.info(f"Dataset split: {split}")
    
    # Upload images in batches
    for batch_start in range(0, len(unlabelled_images), batch_size):
        batch_end = min(batch_start + batch_size, len(unlabelled_images))
        batch = unlabelled_images[batch_start:batch_end]
        
        logger.info(f"Uploading batch {batch_start//batch_size + 1}: "
                   f"images {batch_start + 1}-{batch_end} of {len(unlabelled_images)}")
        
        for idx, image_path in enumerate(batch, batch_start + 1):
            logger.info(f"Uploading {idx}/{len(unlabelled_images)}: {image_path.name}")
            
            try:
                # Upload image without annotations using single_upload
                result = project.single_upload(
                    image_path=str(image_path),
                    split=split,
                    num_retry_uploads=1
                )
                
                stats["uploaded_successfully"] += 1
                logger.debug(f"Successfully uploaded: {image_path.name}")
                logger.debug(f"Upload result: {result}")
                
                # Brief pause to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to upload {image_path.name}: {e}")
                stats["upload_failures"] += 1
                stats["failed_files"].append(str(image_path))
                continue
        
        # Pause between batches
        if batch_end < len(unlabelled_images):
            logger.info(f"Batch completed. Pausing 2 seconds before next batch...")
            time.sleep(2)
    
    stats["processing_time"] = time.time() - start_time
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print upload statistics."""
    logger.info("\n=== UPLOAD STATISTICS ===")
    logger.info(f"Total unlabelled images: {stats['total_images']}")
    logger.info(f"Successfully uploaded: {stats['uploaded_successfully']}")
    logger.info(f"Upload failures: {stats['upload_failures']}")
    logger.info(f"Total processing time: {stats['processing_time']:.1f}s")
    
    if stats['uploaded_successfully'] > 0:
        rate = stats['uploaded_successfully'] / stats['processing_time']
        logger.info(f"Upload rate: {rate:.1f} images/sec")
        
        success_rate = stats['uploaded_successfully'] / stats['total_images'] * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    if stats['failed_files']:
        logger.info(f"\nFailed uploads:")
        for failed_file in stats['failed_files'][:10]:
            logger.info(f"  - {Path(failed_file).name}")
        if len(stats['failed_files']) > 10:
            logger.info(f"  ... and {len(stats['failed_files']) - 10} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload unlabelled images to Roboflow project",
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
        default="container-character-detection",
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
        default=15,
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
    
    logger.info("=== Upload Unlabelled Images to Roboflow ===")
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
        # Create empty labels directory for consistency
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find unlabelled images
        unlabelled_images = find_unlabelled_images(images_dir, labels_dir, logger)
        
        if not unlabelled_images:
            logger.info("No unlabelled images found. All images have corresponding labels.")
            return 0
        
        if args.dry_run:
            logger.info(f"\n✅ Dry run completed!")
            logger.info(f"Found {len(unlabelled_images)} unlabelled images ready for upload")
            logger.info(f"Would upload to '{args.split}' split without annotations")
            
            # Show a few examples
            logger.info("\nExample unlabelled images:")
            for i, img_path in enumerate(unlabelled_images[:10]):
                logger.info(f"  {img_path.name}")
            if len(unlabelled_images) > 10:
                logger.info(f"  ... and {len(unlabelled_images) - 10} more")
            
            return 0
        
        # Upload to Roboflow
        logger.info("\nStarting upload of unlabelled images to Roboflow...")
        
        stats = upload_unlabelled_images(
            args.api_key, args.workspace, args.project,
            unlabelled_images, args.batch_size, args.split, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        if stats["upload_failures"] == 0:
            logger.info(f"\n✅ All unlabelled images uploaded successfully!")
        else:
            logger.warning(f"\n⚠️  Upload completed with {stats['upload_failures']} failures")
        
        logger.info(f"Check your Roboflow project: https://app.roboflow.com/{args.workspace}/{args.project}")
        logger.info("Note: These images can be manually annotated in the Roboflow interface")
        
        return 0 if stats["upload_failures"] == 0 else 1
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())