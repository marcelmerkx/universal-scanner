#!/usr/bin/env python3
"""
Convert OCR labels in-place to YOLO format using filename-based class assignment.

This script converts OCR labels with class 0 to proper YOLO format with numeric 
class IDs (0-35) based on the characters in the filename. The mapping is:
- 0-9: class IDs 0-9 (digits)
- A-Z: class IDs 10-35 (letters)

The script overwrites the original label files in place.

Input/Output: Labels in data/OCR/training_data/11_extra_containers/labels

Usage:
    python3 data/detection/scripts/convert_ocr_labels_to_roboflow.py
    python3 data/detection/scripts/convert_ocr_labels_to_roboflow.py --limit 100
    python3 data/detection/scripts/convert_ocr_labels_to_roboflow.py --backup
"""

import argparse
import logging
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configuration
DEFAULT_LABELS_DIR = Path("data/OCR/training_data/11_extra_containers/labels")
LOG_DIR = Path("data/detection/logs")

# OCR class mapping - character to numeric class ID for YOLO format
# Class indices: 0-9 for digits, 10-35 for A-Z
CHAR_TO_CLASS_ID = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
    'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
    'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
    'Z': 35
}


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "convert_ocr_labels_to_roboflow.log"
    
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


def extract_container_code(filename: str) -> str:
    """
    Extract container code from filename.
    
    Examples:
        "SUDU1234567.jpg" -> "SUDU1234567"
        "HASU4811682_cropped.jpg" -> "HASU4811682"
        "WHLU0409772.jpg" -> "WHLU0409772"
    """
    # Remove extension
    base_name = Path(filename).stem
    
    # Container codes are typically 11 characters (4 letters + 7 digits)
    # Extract the first 11 alphanumeric characters
    match = re.match(r'^([A-Z]{4}\d{7})', base_name.upper())
    if match:
        return match.group(1)
    
    # Fallback: just take first 11 characters
    clean_name = re.sub(r'[_\-\s].*$', '', base_name).upper()
    return clean_name[:11] if len(clean_name) >= 11 else clean_name


def sort_detections_by_x(detections: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    """Sort detections from left to right based on center_x coordinate."""
    return sorted(detections, key=lambda d: d[1])


def convert_label_to_roboflow(
    detections: List[Tuple[int, float, float, float, float]], 
    container_code: str,
    logger: logging.Logger
) -> List[str]:
    """
    Convert detections to Roboflow format with character classes based on container code.
    
    Args:
        detections: List of detections (class_id, center_x, center_y, width, height)
        container_code: Container code extracted from filename
        logger: Logger instance
        
    Returns:
        List of label lines in Roboflow format
    """
    # Sort detections by x-coordinate (left to right)
    sorted_detections = sort_detections_by_x(detections)
    
    # Limit to the number of characters we have in the container code
    num_chars = min(len(sorted_detections), len(container_code))
    
    roboflow_lines = []
    
    for i in range(num_chars):
        det = sorted_detections[i]
        _, center_x, center_y, width, height = det
        
        # Get character from container code
        char = container_code[i].upper()
        
        # Get numeric class ID for YOLO format
        if char in CHAR_TO_CLASS_ID:
            class_id = CHAR_TO_CLASS_ID[char]
            line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            roboflow_lines.append(line)
        else:
            logger.warning(f"Unknown character '{char}' in container code")
    
    return roboflow_lines


def process_label_file(
    label_path: Path,
    backup_dir: Optional[Path],
    logger: logging.Logger
) -> Tuple[bool, str]:
    """
    Process a single label file and convert to Roboflow format in place.
    
    Returns:
        Tuple of (success, error_reason)
    """
    try:
        # Parse label file
        detections = parse_yolo_label(label_path)
        
        if not detections:
            return False, "no_detections"
        
        # Extract container code from filename
        container_code = extract_container_code(label_path.stem)
        
        if len(container_code) < 4:  # Minimum valid container code length
            logger.warning(f"Invalid container code from {label_path.stem}: '{container_code}'")
            return False, "invalid_container_code"
        
        # Convert to Roboflow format
        roboflow_lines = convert_label_to_roboflow(detections, container_code, logger)
        
        if not roboflow_lines:
            return False, "no_valid_conversions"
        
        # Backup original file if requested
        if backup_dir:
            backup_path = backup_dir / label_path.name
            shutil.copy2(label_path, backup_path)
        
        # Overwrite original label file with Roboflow format
        with open(label_path, 'w') as f:
            for line in roboflow_lines:
                f.write(line + '\n')
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Failed to process {label_path.name}: {e}")
        return False, f"exception: {str(e)}"


def process_all_labels(
    labels_dir: Path,
    backup: bool,
    limit: Optional[int],
    logger: logging.Logger
) -> Dict:
    """
    Process all label files and convert to Roboflow format in place.
    
    Returns:
        Processing statistics
    """
    # Create backup directory if requested
    backup_dir = None
    if backup:
        backup_dir = labels_dir.parent / "labels_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backing up original labels to: {backup_dir}")
    
    # Find all label files
    label_files = sorted(list(labels_dir.glob("*.txt")))
    
    if limit:
        label_files = label_files[:limit]
    
    # Statistics
    stats = {
        "total_files": len(label_files),
        "converted": 0,
        "failed": 0,
        "error_reasons": {},
        "character_counts": {}
    }
    
    logger.info(f"Found {len(label_files)} label files to process")
    
    # Process each label file
    for idx, label_path in enumerate(label_files, 1):
        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{len(label_files)} files")
        
        # Process the label
        success, error_reason = process_label_file(
            label_path,
            backup_dir,
            logger
        )
        
        if success:
            stats["converted"] += 1
            
            # Track character distribution
            container_code = extract_container_code(label_path.stem)
            for char in container_code:
                char = char.upper()
                if char in CHAR_TO_CLASS_ID:
                    stats["character_counts"][char] = stats["character_counts"].get(char, 0) + 1
        else:
            stats["failed"] += 1
            stats["error_reasons"][error_reason] = stats["error_reasons"].get(error_reason, 0) + 1
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print processing statistics."""
    logger.info("\n=== PROCESSING STATISTICS ===")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Successfully converted: {stats['converted']}")
    logger.info(f"Failed: {stats['failed']}")
    
    if stats['total_files'] > 0:
        success_rate = stats['converted'] / stats['total_files'] * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Error breakdown
    if stats['error_reasons']:
        logger.info("\nError reasons:")
        for reason, count in sorted(stats['error_reasons'].items()):
            logger.info(f"  {reason}: {count}")
    
    # Character distribution
    if stats['character_counts']:
        logger.info("\nCharacter distribution:")
        total_chars = sum(stats['character_counts'].values())
        for char in sorted(stats['character_counts'].keys()):
            count = stats['character_counts'][char]
            percentage = count / total_chars * 100
            logger.info(f"  {char}: {count:5d} ({percentage:5.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert OCR labels to Roboflow format in place",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="Directory containing OCR label files to convert in place"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original labels before conversion"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Convert OCR Labels to Roboflow Format (In Place) ===")
    logger.info(f"Labels directory: {args.labels_dir}")
    logger.info(f"Create backup: {args.backup}")
    if args.limit:
        logger.info(f"Process limit: {args.limit}")
    
    # Validate input directory
    if not args.labels_dir.exists():
        logger.error(f"Labels directory not found: {args.labels_dir}")
        return 1
    
    # Warn about in-place modification
    if not args.backup:
        logger.warning("⚠️  Labels will be modified in place without backup!")
        logger.info("Use --backup to create a backup before conversion")
    
    try:
        # Process all labels
        stats = process_all_labels(
            args.labels_dir,
            args.backup,
            args.limit,
            logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        logger.info(f"\n✅ Conversion completed!")
        logger.info(f"Labels converted in: {args.labels_dir}")
        
        if args.backup:
            backup_dir = args.labels_dir.parent / "labels_backup"
            logger.info(f"Original labels backed up to: {backup_dir}")
        
        logger.info("\nThe label files are now in Roboflow format and ready for upload.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())