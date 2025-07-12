#!/usr/bin/env python3
"""
Upload images and labels separately to Roboflow project.

This script uploads .jpg files first, then tries to upload .txt files separately,
letting Roboflow handle the matching and annotation parsing.

Usage:
    python3 data/detection/scripts/upload_separate_files_to_roboflow.py
    python3 data/detection/scripts/upload_separate_files_to_roboflow.py --batch-size 10
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
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
    log_file = LOG_DIR / "upload_separate_files_to_roboflow.log"
    
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


def find_image_label_pairs(images_dir: Path, labels_dir: Path, logger: logging.Logger) -> List[Tuple[Path, Path]]:
    """
    Find matching image and label file pairs.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        logger: Logger instance
        
    Returns:
        List of (image_path, label_path) tuples
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = {f.stem: f for f in images_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions}
    
    # Find all label files
    label_files = {f.stem: f for f in labels_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() == '.txt'}
    
    # Match images with labels
    pairs = []
    for stem in image_files:
        if stem in label_files:
            pairs.append((image_files[stem], label_files[stem]))
        else:
            logger.warning(f"No label file found for image: {image_files[stem].name}")
    
    logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")
    logger.info(f"Matched {len(pairs)} image-label pairs")
    
    return pairs


def upload_single_files_to_roboflow(api_key: str, workspace: str, project_id: str,
                                   image_label_pairs: List[Tuple[Path, Path]], 
                                   batch_size: int, split: str, logger: logging.Logger) -> Dict:
    """
    Upload images with corresponding YOLO labels using single_upload.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Roboflow project ID
        image_label_pairs: List of (image_path, label_path) tuples
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
        "total_pairs": len(image_label_pairs),
        "uploaded_successfully": 0,
        "upload_failures": 0,
        "processing_time": 0.0,
        "failed_files": []
    }
    
    start_time = time.time()
    
    logger.info(f"=== Uploading {len(image_label_pairs)} image-label pairs ===")
    logger.info(f"Using single_upload with image_path + annotation_path")
    logger.info(f"Dataset split: {split}")
    
    # Upload image-label pairs
    for batch_start in range(0, len(image_label_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(image_label_pairs))
        batch = image_label_pairs[batch_start:batch_end]
        
        logger.info(f"Uploading batch {batch_start//batch_size + 1}: "
                   f"pairs {batch_start + 1}-{batch_end} of {len(image_label_pairs)}")
        
        for idx, (image_path, label_path) in enumerate(batch, batch_start + 1):
            logger.info(f"Uploading {idx}/{len(image_label_pairs)}: {image_path.name} + {label_path.name}")
            
            try:
                # Upload image with corresponding YOLO label
                result = project.single_upload(
                    image_path=str(image_path),
                    annotation_path=str(label_path),
                    split=split,
                    num_retry_uploads=1
                )
                
                stats["uploaded_successfully"] += 1
                logger.debug(f"Successfully uploaded: {image_path.name}")
                logger.debug(f"Upload result: {result}")
                
                # Brief pause to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to upload {image_path.name}: {e}")
                stats["upload_failures"] += 1
                stats["failed_files"].append(str(image_path))
                continue
        
        # Pause between batches
        if batch_end < len(image_label_pairs):
            logger.info(f"Batch completed. Pausing 3 seconds before next batch...")
            time.sleep(3)
    
    stats["processing_time"] = time.time() - start_time
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print upload statistics."""
    logger.info("\n=== UPLOAD STATISTICS ===")
    logger.info(f"Total image-label pairs: {stats['total_pairs']}")
    logger.info(f"Successfully uploaded: {stats['uploaded_successfully']}")
    logger.info(f"Upload failures: {stats['upload_failures']}")
    logger.info(f"Total processing time: {stats['processing_time']:.1f}s")
    
    if stats['uploaded_successfully'] > 0:
        rate = stats['uploaded_successfully'] / stats['processing_time']
        logger.info(f"Upload rate: {rate:.1f} pairs/sec")
        
        success_rate = stats['uploaded_successfully'] / stats['total_pairs'] * 100
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
        description="Upload images and labels separately to Roboflow project",
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
        default=10,
        help="Number of files to upload in each batch"
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
    
    logger.info("=== Upload Images and Labels to Roboflow using single_upload ===")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels directory: {labels_dir}")
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Project: {args.project}")
    logger.info(f"Dataset split: {args.split}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("\nMethod: single_upload with image_path + annotation_path")
    
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
        logger.error(f"Labels directory not found: {labels_dir}")
        return 1
    
    try:
        # Find image-label pairs
        image_label_pairs = find_image_label_pairs(images_dir, labels_dir, logger)
        
        if not image_label_pairs:
            logger.error("No matching image-label pairs found. Nothing to upload.")
            return 1
        
        if args.dry_run:
            logger.info(f"\n✅ Dry run completed!")
            logger.info(f"Found {len(image_label_pairs)} image-label pairs ready for upload")
            logger.info(f"Would upload using single_upload to '{args.split}' split")
            
            # Show a few examples
            logger.info("\nExample pairs to upload:")
            for i, (img_path, lbl_path) in enumerate(image_label_pairs[:5]):
                logger.info(f"  {img_path.name} + {lbl_path.name}")
            if len(image_label_pairs) > 5:
                logger.info(f"  ... and {len(image_label_pairs) - 5} more pairs")
            
            return 0
        
        # Upload to Roboflow
        logger.info("\nStarting upload to Roboflow...")
        
        stats = upload_single_files_to_roboflow(
            args.api_key, args.workspace, args.project,
            image_label_pairs, args.batch_size, args.split, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        if stats["upload_failures"] == 0:
            logger.info(f"\n✅ All uploads completed successfully!")
        else:
            logger.warning(f"\n⚠️  Upload completed with {stats['upload_failures']} failures")
        
        logger.info(f"Check your Roboflow project: https://app.roboflow.com/{args.workspace}/{args.project}")
        logger.info("Note: It may take a few minutes for annotations to appear in the Roboflow interface")
        
        return 0 if stats["upload_failures"] == 0 else 1
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())