#!/usr/bin/env python3
"""
Convert YOLO labels from numeric class IDs to class names for Roboflow upload.

This script converts label files that use numeric class IDs (0-35) to use
actual class names ('0'-'9', 'A'-'Z') which Roboflow can properly interpret.

Usage:
    python3 data/detection/scripts/convert_labels_for_roboflow.py
    python3 data/detection/scripts/convert_labels_for_roboflow.py --input-dir data/detection/training_data/03_ocr_generated/labels
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict

# OCR class mapping - maps class IDs to actual class names
OCR_CLASS_NAMES = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z'
}


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def convert_label_file(input_path: Path, output_path: Path, logger: logging.Logger) -> bool:
    """
    Convert a single label file from numeric IDs to class names.
    
    Args:
        input_path: Path to input label file
        output_path: Path to output label file
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        converted_lines = []
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    logger.warning(f"Invalid format in {input_path.name} line {line_num}: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    
                    # Convert numeric ID to class name
                    if class_id in OCR_CLASS_NAMES:
                        class_name = OCR_CLASS_NAMES[class_id]
                        # Reconstruct line with class name instead of ID
                        converted_line = f"{class_name} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                        converted_lines.append(converted_line)
                    else:
                        logger.warning(f"Unknown class ID {class_id} in {input_path.name}")
                        # Keep original line if class ID is unknown
                        converted_lines.append(line)
                        
                except ValueError as e:
                    logger.warning(f"Could not parse class ID in {input_path.name} line {line_num}: {e}")
                    continue
        
        # Write converted labels
        with open(output_path, 'w') as f:
            for line in converted_lines:
                f.write(line + '\n')
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {input_path.name}: {e}")
        return False


def convert_all_labels(input_dir: Path, output_dir: Path, logger: logging.Logger) -> Dict:
    """
    Convert all label files in a directory.
    
    Args:
        input_dir: Directory containing label files with numeric IDs
        output_dir: Directory where converted labels will be saved
        logger: Logger instance
        
    Returns:
        Conversion statistics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all label files
    label_files = sorted(list(input_dir.glob("*.txt")))
    
    stats = {
        "total_files": len(label_files),
        "converted": 0,
        "failed": 0
    }
    
    logger.info(f"Found {len(label_files)} label files to convert")
    
    for idx, label_path in enumerate(label_files, 1):
        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{len(label_files)} files")
        
        output_path = output_dir / label_path.name
        
        if convert_label_file(label_path, output_path, logger):
            stats["converted"] += 1
        else:
            stats["failed"] += 1
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels from numeric IDs to class names for Roboflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/detection/training_data/03_ocr_generated/labels"),
        help="Directory containing label files with numeric class IDs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detection/training_data/03_ocr_generated/labels_roboflow"),
        help="Directory where converted labels will be saved"
    )
    
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Also copy corresponding images to a parallel directory"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Convert Labels for Roboflow ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    try:
        # Convert labels
        stats = convert_all_labels(args.input_dir, args.output_dir, logger)
        
        # Print statistics
        logger.info("\n=== CONVERSION STATISTICS ===")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Successfully converted: {stats['converted']}")
        logger.info(f"Failed: {stats['failed']}")
        
        if stats['converted'] > 0:
            success_rate = stats['converted'] / stats['total_files'] * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Copy images if requested
        if args.copy_images and stats['converted'] > 0:
            logger.info("\n=== Copying Images ===")
            
            # Determine source images directory
            images_dir = args.input_dir.parent / "images"
            if not images_dir.exists():
                logger.error(f"Images directory not found: {images_dir}")
                return 1
            
            # Create output images directory
            output_images_dir = args.output_dir.parent / "images_roboflow"
            output_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images that have converted labels
            copied = 0
            for label_file in args.output_dir.glob("*.txt"):
                # Find corresponding image
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = images_dir / f"{label_file.stem}{ext}"
                    if image_path.exists():
                        output_image_path = output_images_dir / image_path.name
                        shutil.copy2(image_path, output_image_path)
                        copied += 1
                        break
            
            logger.info(f"Copied {copied} images to {output_images_dir}")
        
        logger.info(f"\n✅ Conversion completed!")
        logger.info(f"Converted labels saved to: {args.output_dir}")
        
        if args.copy_images:
            logger.info("\nYou can now upload to Roboflow using:")
            logger.info(f"  --labelled-dir {args.output_dir.parent}")
            logger.info("  (This will use the images_roboflow and labels_roboflow subdirectories)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())