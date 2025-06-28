#!/usr/bin/env python3
"""
Test Roboflow Model Inference on Container Dataset

This script downloads a trained Roboflow model and runs inference on container images
to evaluate detection performance and accuracy.

Requirements:
    pip install roboflow opencv-python pillow

Usage:
    # Test on small subset (10 images)
    python3 detection/scripts/test_roboflow_inference.py --limit 10
    
    # Test on larger subset with confidence threshold
    python3 detection/scripts/test_roboflow_inference.py --limit 100 --confidence 0.5
    
    # Full dataset test
    python3 detection/scripts/test_roboflow_inference.py

The script:
1. Downloads the Roboflow model using API key
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
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Roboflow imports
try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed. Run: pip install roboflow")
    exit(1)

# Configuration
ROBOFLOW_PROJECT_ID = "container_code_detection_pretrain_faster-mrjxf-kxw1i"
INPUT_DIR = Path("detection/training_data/00_raw/container_code_tbd")
RESULTS_DIR = Path("detection/results")
LOG_DIR = Path("detection/logs")

# Set up logging
def setup_logging():
    """Configure logging for inference testing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "roboflow_inference_test.log"
    
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

def load_roboflow_model(api_key: str, project_id: str, logger: logging.Logger):
    """
    Download and load Roboflow model.
    
    Args:
        api_key: Roboflow API key
        project_id: Roboflow project identifier
        logger: Logger instance
        
    Returns:
        Roboflow model instance
    """
    try:
        logger.info(f"Connecting to Roboflow with project: {project_id}")
        rf = Roboflow(api_key=api_key)
        
        # Parse project ID format: "project-name-version"
        if "-" in project_id:
            project_name = project_id.rsplit("-", 1)[0]
            version = project_id.rsplit("-", 1)[1]
        else:
            project_name = project_id
            version = "1"  # Default version
        
        logger.info(f"Loading project: {project_name}, version: {version}")
        project = rf.workspace().project(project_name)
        model = project.version(version).model
        
        logger.info("✓ Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Roboflow model: {e}")
        logger.info("Check your API key and project ID")
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

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes with keys 'x', 'y', 'width', 'height'
        
    Returns:
        IoU value between 0 and 1
    """
    # Convert to x1, y1, x2, y2 format
    x1_1, y1_1 = box1['x'] - box1['width']/2, box1['y'] - box1['height']/2
    x2_1, y2_1 = box1['x'] + box1['width']/2, box1['y'] + box1['height']/2
    
    x1_2, y1_2 = box2['x'] - box2['width']/2, box2['y'] - box2['height']/2
    x2_2, y2_2 = box2['x'] + box2['width']/2, box2['y'] + box2['height']/2
    
    # Calculate intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0
    
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def draw_bounding_boxes(image_path: Path, detections: List[Dict], output_path: Path, logger: logging.Logger):
    """
    Draw bounding boxes on image and save result.
    
    Args:
        image_path: Path to original image
        detections: List of detection results from Roboflow
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
            # Get bounding box coordinates (center format to corner format)
            x_center = detection['x']
            y_center = detection['y']
            width = detection['width']
            height = detection['height']
            
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # Draw rectangle
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class', 'unknown')
            
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
        model: Loaded Roboflow model
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
            "project_id": ROBOFLOW_PROJECT_ID,
            "confidence_threshold": confidence_threshold,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
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
            predictions = model.predict(str(image_path), confidence=confidence_threshold).json()
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Extract detections
            detections = predictions.get('predictions', [])
            
            # Calculate metrics for this image
            image_result = {
                "filename": image_path.name,
                "expected_code": extract_container_code_from_filename(image_path.name),
                "detections": detections,
                "num_detections": len(detections),
                "inference_time": inference_time,
                "max_confidence": max([d.get('confidence', 0.0) for d in detections], default=0.0)
            }
            
            results["images"].append(image_result)
            
            # Update counters
            if detections:
                results["summary"]["images_with_detections"] += 1
                total_detections += len(detections)
                for det in detections:
                    total_confidence += det.get('confidence', 0.0)
            
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
            all_confidences.append(det.get('confidence', 0.0))
    
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
        }
    }
    
    # Log key metrics
    logger.info(f"Detection Rate: {detection_rate:.1%}")
    logger.info(f"Avg Detections per Image: {performance['avg_detections_per_image']:.1f}")
    logger.info(f"Avg Confidence: {performance['confidence_stats']['mean']:.3f}")
    logger.info(f"Avg Inference Time: {performance['timing_stats']['avg_inference_time']:.3f}s")
    
    return performance

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Roboflow model inference on container dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Get API key
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        logger.error("Roboflow API key required. Set ROBOFLOW_API_KEY env var or use --api-key")
        return 1
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Roboflow Model Inference Test ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Image limit: {args.limit}")
    logger.info(f"Confidence threshold: {args.confidence}")
    
    try:
        # Load model
        model = load_roboflow_model(api_key, ROBOFLOW_PROJECT_ID, logger)
        
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
        
        if performance['detection_rate'] < 0.5:
            logger.warning("⚠ Low detection rate - consider adjusting confidence threshold or model")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 