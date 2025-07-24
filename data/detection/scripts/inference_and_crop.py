#!/usr/bin/env python3
"""
Run inference on container door images and crop detected regions.

This script:
1. Loads a PyTorch YOLO detection model
2. Runs inference on images from specified input directory
3. Crops detected regions and saves them to output directory organized by class
4. Preserves original filenames for cropped images
5. Copies images with no detections to a separate folder for analysis

Usage:
    # Basic test with 10 images
    python3 data/detection/scripts/inference_and_crop.py --limit 10
    
    # Custom input/output directories
    python3 data/detection/scripts/inference_and_crop.py \
        --input-dir /path/to/your/images \
        --output-dir /path/to/output \
        --limit 10
    
    # Full run with all parameters
    python3 data/detection/scripts/inference_and_crop.py \
        --input-dir data/OCR/horizontal \
        --output-dir data/OCR/horizontal_crops \
        --model-path data/detection/models/your-model.pt \
        --confidence 0.3 \
        --padding 15

Parameters:
    --input-dir: Directory containing input images (default: data/detection/containerdoors/images)
    --output-dir: Base directory for output crops (default: data/detection/training_data/00_raw)
    --model-path: Path to YOLO .pt model file (default: data/detection/models/detection_320_grayscale_tilted-09-07-2025.pt)
    --limit: Maximum number of images to process, useful for testing (default: None, process all)
    --confidence: Confidence threshold for detections (default: 0.3)
    --padding: Pixels to add around detections when cropping (default: 10)

Output structure:
    output-dir/
    ├── {class_name}/       # Cropped detections organized by class
    │   ├── image1.jpg
    │   └── image2_1.jpg    # Multiple detections get indexed
    └── no_detection/       # Images where no detections were found
        └── image3.jpg
"""

import argparse
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed. Run: pip install ultralytics")
    exit(1)

# Configuration
DEFAULT_MODEL_PATH = Path("data/detection/models/detection_320_grayscale_tilted-09-07-2025.pt")
INPUT_DIR = Path("data/detection/containerdoors/images")
OUTPUT_BASE_DIR = Path("data/detection/training_data/00_raw")
LOG_DIR = Path("data/detection/logs")

# Set up logging
def setup_logging():
    """Configure logging for inference and cropping."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "inference_and_crop.log"
    
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


def preprocess_image_for_model(image: np.ndarray, target_size: int = 320) -> Tuple[np.ndarray, float, int, int]:
    """
    Preprocess image to 320x320 grayscale with aspect-ratio preserving padding.
    
    Args:
        image: Original BGR image array
        target_size: Target square size (default 320)
        
    Returns:
        Tuple of (preprocessed_image, scale_factor, x_offset, y_offset)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    
    # Calculate scale to fit within target_size while preserving aspect ratio
    scale = min(target_size / width, target_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(gray, (new_width, new_height))
    
    # Create padded image with white background
    padded = np.full((target_size, target_size), 255, dtype=np.uint8)
    
    # Calculate padding offsets (center the image)
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Convert back to 3-channel for YOLO
    preprocessed = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    
    return preprocessed, scale, x_offset, y_offset


def convert_coordinates_to_original(bbox_320: List[int], scale: float, x_offset: int, y_offset: int, 
                                   orig_width: int, orig_height: int) -> List[int]:
    """
    Convert bounding box coordinates from 320x320 padded image back to original image coordinates.
    
    Args:
        bbox_320: Bounding box [x1, y1, x2, y2] in 320x320 image
        scale: Scale factor used during preprocessing
        x_offset: X offset used during padding
        y_offset: Y offset used during padding
        orig_width: Original image width
        orig_height: Original image height
        
    Returns:
        Bounding box [x1, y1, x2, y2] in original image coordinates
    """
    x1_320, y1_320, x2_320, y2_320 = bbox_320
    
    # Convert back to original coordinates
    x1_orig = int((x1_320 - x_offset) / scale)
    y1_orig = int((y1_320 - y_offset) / scale)
    x2_orig = int((x2_320 - x_offset) / scale)
    y2_orig = int((y2_320 - y_offset) / scale)
    
    # Clamp to image bounds
    x1_orig = max(0, min(orig_width, x1_orig))
    y1_orig = max(0, min(orig_height, y1_orig))
    x2_orig = max(0, min(orig_width, x2_orig))
    y2_orig = max(0, min(orig_height, y2_orig))
    
    return [x1_orig, y1_orig, x2_orig, y2_orig]


def crop_detection(image: np.ndarray, bbox: List[int], padding: int = 10) -> np.ndarray:
    """
    Crop a detection from an image with optional padding.
    
    Args:
        image: Original image array
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding to add around the detection
        
    Returns:
        Cropped image array
    """
    height, width = image.shape[:2]
    
    # Extract coordinates and add padding
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    
    # Crop the image
    cropped = image[y1:y2, x1:x2]
    
    return cropped


def save_cropped_detection(cropped_image: np.ndarray, class_name: str, 
                          original_filename: str, detection_idx: int,
                          output_base_dir: Path, logger: logging.Logger) -> Path:
    """
    Save a cropped detection to the appropriate class directory.
    
    Args:
        cropped_image: Cropped image array
        class_name: Detection class name
        original_filename: Original image filename
        detection_idx: Index of detection in the image
        output_base_dir: Base output directory
        logger: Logger instance
        
    Returns:
        Path to saved file
    """
    # Create class directory if it doesn't exist
    class_dir = output_base_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    # If there's only one detection, use original filename
    # If multiple detections, append index
    base_name = Path(original_filename).stem
    if detection_idx == 0:
        output_filename = f"{base_name}.jpg"
    else:
        output_filename = f"{base_name}_{detection_idx}.jpg"
    
    output_path = class_dir / output_filename
    
    # Check if file already exists
    if output_path.exists():
        # Add timestamp to make unique
        timestamp = int(time.time())
        output_filename = f"{base_name}_{detection_idx}_{timestamp}.jpg"
        output_path = class_dir / output_filename
    
    # Save the cropped image
    cv2.imwrite(str(output_path), cropped_image)
    
    return output_path


def copy_no_detection_image(image_path: Path, output_base_dir: Path, logger: logging.Logger) -> Path:
    """
    Copy image with no detections to no_detection subfolder.
    
    Args:
        image_path: Path to original image
        output_base_dir: Base output directory
        logger: Logger instance
        
    Returns:
        Path to copied file
    """
    # Create no_detection directory if it doesn't exist
    no_detection_dir = output_base_dir / "no_detection"
    no_detection_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy image to no_detection folder
    output_path = no_detection_dir / image_path.name
    shutil.copy2(image_path, output_path)
    
    logger.debug(f"Copied no-detection image to {output_path}")
    
    return output_path


def process_images(model, image_dir: Path, output_base_dir: Path,
                  limit: Optional[int], confidence_threshold: float,
                  logger: logging.Logger) -> Dict:
    """
    Process images through the model and save cropped detections.
    
    Args:
        model: Loaded YOLO model
        image_dir: Directory containing input images
        output_base_dir: Base directory for output
        limit: Maximum number of images to process
        confidence_threshold: Minimum confidence for detections
        logger: Logger instance
        
    Returns:
        Dictionary with processing statistics
    """
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in image_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if limit:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Statistics
    stats = {
        "total_images": len(image_files),
        "images_processed": 0,
        "total_detections": 0,
        "detections_saved": 0,
        "detections_per_class": {},
        "failed_images": 0,
        "images_no_detections": 0,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                stats["failed_images"] += 1
                continue
            
            orig_height, orig_width = image.shape[:2]
            
            # Preprocess image to 320x320 grayscale with padding
            preprocessed, scale, x_offset, y_offset = preprocess_image_for_model(image)
            
            # Save preprocessed image temporarily for inference
            temp_path = f"/tmp/preprocessed_{image_path.stem}.jpg"
            cv2.imwrite(temp_path, preprocessed)
            
            # Run inference on preprocessed image
            results = model(temp_path, conf=confidence_threshold, verbose=False)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            # Process detections
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Track detections per class for this image
                class_counts = {}
                detections_found = False
                
                for i in range(len(boxes)):
                    # Get detection info (coordinates are in 320x320 space)
                    x1_320, y1_320, x2_320, y2_320 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Convert coordinates back to original image space
                    bbox_320 = [int(x1_320), int(y1_320), int(x2_320), int(y2_320)]
                    bbox_orig = convert_coordinates_to_original(
                        bbox_320, scale, x_offset, y_offset, orig_width, orig_height
                    )
                    
                    # Count detections per class for unique indexing
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    detection_idx = class_counts[class_name]
                    class_counts[class_name] += 1
                    
                    # Crop detection from original image
                    cropped = crop_detection(image, bbox_orig)
                    
                    logger.debug(f"Detection {class_name}: 320x320 bbox={bbox_320}, "
                               f"original bbox={bbox_orig}, confidence={confidence:.3f}")
                    
                    # Save cropped detection
                    output_path = save_cropped_detection(
                        cropped, class_name, image_path.name, 
                        detection_idx, output_base_dir, logger
                    )
                    
                    logger.debug(f"Saved {class_name} detection to {output_path}")
                    
                    # Update statistics
                    stats["total_detections"] += 1
                    stats["detections_saved"] += 1
                    if class_name not in stats["detections_per_class"]:
                        stats["detections_per_class"][class_name] = 0
                    stats["detections_per_class"][class_name] += 1
                    detections_found = True
                
                # Check if no detections were found above threshold
                if not detections_found:
                    stats["images_no_detections"] += 1
                    logger.debug(f"No detections above confidence {confidence_threshold} for {image_path.name}")
                    # Copy to no_detection folder for analysis
                    copy_no_detection_image(image_path, output_base_dir, logger)
            
            else:
                # No detections at all
                stats["images_no_detections"] += 1
                logger.debug(f"No detections found for {image_path.name}")
                # Copy to no_detection folder for analysis
                copy_no_detection_image(image_path, output_base_dir, logger)
            
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
    logger.info(f"Images with no detections: {stats['images_no_detections']}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Detections saved: {stats['detections_saved']}")
    logger.info(f"Processing time: {stats['processing_time']:.1f}s")
    
    if stats['images_processed'] > 0:
        rate = stats['images_processed'] / stats['processing_time']
        logger.info(f"Processing rate: {rate:.1f} images/sec")
        
        avg_detections = stats['total_detections'] / stats['images_processed']
        logger.info(f"Average detections per image: {avg_detections:.1f}")
    
    logger.info("\nDetections per class:")
    for class_name, count in sorted(stats['detections_per_class'].items()):
        logger.info(f"  {class_name}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference and crop detections from container door images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO model (.pt file)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE_DIR,
        help="Base directory for output (crops will be saved to subdirectories by class)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (useful for testing)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding to add around detections when cropping"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Validate paths
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Inference and Crop Script ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Image limit: {args.limit if args.limit else 'None (process all)'}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Crop padding: {args.padding}")
    
    try:
        # Load model
        model = load_yolo_model(args.model_path, logger)
        
        # Process images
        logger.info("\nStarting image processing...")
        stats = process_images(
            model, args.input_dir, args.output_dir,
            args.limit, args.confidence, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        logger.info("\n✅ Processing completed successfully!")
        logger.info(f"Cropped images saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())