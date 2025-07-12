#!/usr/bin/env python3
"""
Run inference on container door images and crop detected regions.

This script:
1. Loads a PyTorch YOLO detection model
2. Runs inference on images from data/detection/containerdoors
3. Crops detected regions and saves them to data/detection/training_data/00_raw/{class_name}
4. Preserves original filenames for cropped images

Usage:
    python3 data/detection/scripts/inference_and_crop.py --limit 10
    python3 data/detection/scripts/inference_and_crop.py --confidence 0.5
    python3 data/detection/scripts/inference_and_crop.py --model-path path/to/model.pt
"""

import argparse
import logging
import time
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
INPUT_DIR = Path("data/detection/containerdoors")
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
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
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
            
            # Run inference
            results = model(str(image_path), conf=confidence_threshold, verbose=False)
            
            # Process detections
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Track detections per class for this image
                class_counts = {}
                
                for i in range(len(boxes)):
                    # Get detection info
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Count detections per class for unique indexing
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    detection_idx = class_counts[class_name]
                    class_counts[class_name] += 1
                    
                    # Crop detection
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    cropped = crop_detection(image, bbox)
                    
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
        default=0.25,
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