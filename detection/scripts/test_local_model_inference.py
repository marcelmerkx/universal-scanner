#!/usr/bin/env python3
"""
Test Local YOLO Model Inference on Container Dataset

This script loads a local PyTorch YOLO model and runs inference on container images
to evaluate detection performance and accuracy.

Requirements:
    pip install ultralytics opencv-python pillow matplotlib

Usage:
    # Test on small subset (10 images)
    python3 detection/scripts/test_local_model_inference.py --limit 10
    
    # Test with custom confidence threshold
    python3 detection/scripts/test_local_model_inference.py --limit 50 --confidence 0.5
    
    # Use custom model path
    python3 detection/scripts/test_local_model_inference.py --model-path path/to/model.pt

The script:
1. Loads the local PyTorch YOLO model
2. Runs inference on container images
3. Saves results with bounding box data
4. Generates performance metrics and timing analysis
5. Creates visualization-ready data for Jupyter notebook

Output:
- detection/results/inference_results.json
- detection/results/performance_metrics.json  
- detection/results/sample_images/ (with bounding boxes drawn)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed. Run: pip install ultralytics")
    exit(1)

# Configuration
DEFAULT_MODEL_PATH = Path("detection/models/roboflow/weights.pt")
INPUT_DIR = Path("detection/training_data/00_raw/container_code_tbd")
RESULTS_DIR = Path("detection/results")
LOG_DIR = Path("detection/logs")

# Set up logging
def setup_logging():
    """Configure logging for inference testing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "local_model_inference_test.log"
    
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

def extract_container_code_from_filename(filename: str) -> str:
    """
    Extract expected container code from filename.
    
    Args:
        filename: Image filename (e.g., "CAIU1234567.jpg")
        
    Returns:
        Container code without extension
    """
    return Path(filename).stem

def draw_bounding_boxes(image_path: Path, detections: List[Dict], output_path: Path, logger: logging.Logger):
    """
    Draw bounding boxes on image and save result.
    
    Args:
        image_path: Path to original image
        detections: List of detection results from YOLO
        output_path: Path to save annotated image
        logger: Logger instance
    """
    try:
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw each detection
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Color based on confidence (red = low, green = high)
            if confidence > 0.8:
                color = "green"
            elif confidence > 0.5:
                color = "orange"
            else:
                color = "red"
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1-20), label, fill=color, font=font)
        
        # Save annotated image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        
    except Exception as e:
        logger.error(f"Failed to draw bounding boxes for {image_path}: {e}")

def run_inference_batch(model, image_dir: Path, limit: Optional[int], confidence_threshold: float, 
                       logger: logging.Logger) -> Dict:
    """
    Run inference on a batch of images.
    
    Args:
        model: Loaded YOLO model
        image_dir: Directory containing images
        limit: Maximum number of images to process
        confidence_threshold: Minimum confidence for detections
        logger: Logger instance
        
    Returns:
        Dictionary with inference results and metrics
    """
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if limit:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Results storage
    results = {
        "model_info": {
            "model_path": str(DEFAULT_MODEL_PATH),
            "confidence_threshold": confidence_threshold,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_classes": list(model.names.values())
        },
        "images": [],
        "summary": {
            "total_images": len(image_files),
            "images_with_detections": 0,
            "total_detections": 0,
            "avg_confidence": 0.0,
            "avg_inference_time": 0.0
        }
    }
    
    total_inference_time = 0.0
    total_confidence = 0.0
    total_detections = 0
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        
        try:
            # Run inference
            start_time = time.time()
            results_yolo = model(str(image_path), conf=confidence_threshold, verbose=False)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Extract detections
            detections = []
            if len(results_yolo) > 0 and results_yolo[0].boxes is not None:
                boxes = results_yolo[0].boxes
                for i in range(len(boxes)):
                    # Get box coordinates (xyxy format)
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
            
            # Calculate metrics for this image
            image_result = {
                "filename": image_path.name,
                "expected_code": extract_container_code_from_filename(image_path.name),
                "detections": detections,
                "num_detections": len(detections),
                "inference_time": inference_time,
                "max_confidence": max([d['confidence'] for d in detections], default=0.0)
            }
            
            results["images"].append(image_result)
            
            # Update counters
            if detections:
                results["summary"]["images_with_detections"] += 1
                total_detections += len(detections)
                for det in detections:
                    total_confidence += det['confidence']
            
            # Save annotated image for first 10 results
            if idx <= 10:
                output_path = RESULTS_DIR / "sample_images" / f"annotated_{image_path.name}"
                draw_bounding_boxes(image_path, detections, output_path, logger)
            
            # Progress update
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{len(image_files)} images processed")
                
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            continue
    
    # Calculate final metrics
    if total_detections > 0:
        results["summary"]["avg_confidence"] = total_confidence / total_detections
    results["summary"]["total_detections"] = total_detections
    results["summary"]["avg_inference_time"] = total_inference_time / len(image_files)
    
    return results

def analyze_performance(results: Dict, logger: logging.Logger) -> Dict:
    """
    Analyze model performance and generate metrics.
    
    Args:
        results: Inference results dictionary
        logger: Logger instance
        
    Returns:
        Performance analysis dictionary
    """
    logger.info("Analyzing model performance...")
    
    images = results["images"]
    total_images = len(images)
    
    # Detection rate analysis
    images_with_detections = sum(1 for img in images if img["num_detections"] > 0)
    detection_rate = images_with_detections / total_images if total_images > 0 else 0
    
    # Confidence distribution
    all_confidences = []
    for img in images:
        for det in img["detections"]:
            all_confidences.append(det['confidence'])
    
    # Class distribution
    class_counts = {}
    for img in images:
        for det in img["detections"]:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Performance metrics
    performance = {
        "detection_rate": detection_rate,
        "avg_detections_per_image": sum(img["num_detections"] for img in images) / total_images,
        "confidence_stats": {
            "mean": np.mean(all_confidences) if all_confidences else 0.0,
            "median": np.median(all_confidences) if all_confidences else 0.0,
            "std": np.std(all_confidences) if all_confidences else 0.0,
            "min": min(all_confidences) if all_confidences else 0.0,
            "max": max(all_confidences) if all_confidences else 0.0
        },
        "timing_stats": {
            "avg_inference_time": results["summary"]["avg_inference_time"],
            "total_time": sum(img["inference_time"] for img in images)
        },
        "class_distribution": class_counts
    }
    
    # Log key metrics
    logger.info(f"Detection Rate: {detection_rate:.1%}")
    logger.info(f"Avg Detections per Image: {performance['avg_detections_per_image']:.1f}")
    logger.info(f"Avg Confidence: {performance['confidence_stats']['mean']:.3f}")
    logger.info(f"Avg Inference Time: {performance['timing_stats']['avg_inference_time']:.3f}s")
    logger.info(f"Class Distribution: {class_counts}")
    
    return performance

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test local YOLO model inference on container dataset",
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
        help="Directory containing container images"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of images to process"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Validate model file
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Local YOLO Model Inference Test ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Image limit: {args.limit}")
    logger.info(f"Confidence threshold: {args.confidence}")
    
    try:
        # Load model
        model = load_yolo_model(args.model_path, logger)
        
        # Run inference
        logger.info("Starting batch inference...")
        start_time = time.time()
        
        results = run_inference_batch(
            model, args.input_dir, args.limit, args.confidence, logger
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch inference completed in {elapsed_time:.1f}s")
        
        # Analyze performance
        performance = analyze_performance(results, logger)
        
        # Save results
        results_file = RESULTS_DIR / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        performance_file = RESULTS_DIR / "performance_metrics.json"
        with open(performance_file, 'w') as f:
            json.dump(performance, f, indent=2)
        
        logger.info(f"\n✓ Results saved to:")
        logger.info(f"  - {results_file}")
        logger.info(f"  - {performance_file}")
        logger.info(f"  - {RESULTS_DIR}/sample_images/ (annotated images)")
        
        # Summary
        logger.info("\n=== PERFORMANCE SUMMARY ===")
        logger.info(f"Images processed: {results['summary']['total_images']}")
        logger.info(f"Detection rate: {performance['detection_rate']:.1%}")
        logger.info(f"Avg confidence: {performance['confidence_stats']['mean']:.3f}")
        logger.info(f"Avg inference time: {performance['timing_stats']['avg_inference_time']:.3f}s")
        logger.info(f"Classes detected: {list(performance['class_distribution'].keys())}")
        
        if performance['detection_rate'] < 0.5:
            logger.warning("⚠ Low detection rate - consider adjusting confidence threshold or model")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 