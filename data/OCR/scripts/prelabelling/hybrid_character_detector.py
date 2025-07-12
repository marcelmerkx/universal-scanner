"""
Hybrid Character Detection for Container Codes

This script uses multiple detection strategies to find character-looking areas:
1. YOLO model detection
2. OpenCV contour detection  
3. Template matching (if needed)
4. Fallback grid estimation

The goal is to identify roughly 11 character regions for manual correction.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json

import cv2
import numpy as np
from ultralytics import YOLO

# Import config
try:
    from config_template import INPUT_DIR, OUTPUT_DIR
    MODEL_PATH = "data/OCR/models/best-OCR-18-06-25.pt"
except ImportError:
    MODEL_PATH = "data/OCR/models/best-OCR-18-06-25.pt"
    INPUT_DIR = "data/OCR/cutouts"
    OUTPUT_DIR = "data/OCR/labels"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_CHARS = 11
CONF_THRESHOLD = 0.05  # Very low for YOLO
MIN_CONTOUR_AREA = 50
MAX_CONTOUR_AREA = 2000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('hybrid_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection Strategies
# ---------------------------------------------------------------------------

def yolo_detection(image: np.ndarray, model: YOLO, conf_threshold: float = CONF_THRESHOLD) -> List[Tuple[int, int, int, int]]:
    """Strategy 1: Use YOLO model to detect characters"""
    try:
        results = model(image, conf=conf_threshold, verbose=False)[0]
        if results.boxes is None or results.boxes.data.numel() == 0:
            return []
        
        boxes = results.boxes.xyxy.cpu().numpy()
        # Convert to (x1, y1, x2, y2) integers
        return [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
    except Exception as e:
        logger.warning(f"YOLO detection failed: {e}")
        return []


def contour_detection(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Strategy 2: Use OpenCV contours to find character-like regions"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        # Try multiple threshold methods
        methods = [
            lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda g: cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)[1]
        ]
        
        all_boxes = []
        
        for method in methods:
            try:
                binary = method(gray)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                boxes = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Filter by aspect ratio (characters are usually taller than wide)
                        aspect_ratio = h / w if w > 0 else 0
                        if 0.5 <= aspect_ratio <= 4.0:  # Allow some variation
                            boxes.append((x, y, x + w, y + h))
                
                all_boxes.extend(boxes)
            except Exception as e:
                logger.debug(f"Threshold method failed: {e}")
                continue
        
        # Remove duplicates (boxes that are very close)
        unique_boxes = []
        for box in all_boxes:
            is_duplicate = False
            for existing in unique_boxes:
                if boxes_overlap(box, existing, threshold=0.5):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boxes.append(box)
        
        return unique_boxes
        
    except Exception as e:
        logger.warning(f"Contour detection failed: {e}")
        return []


def grid_estimation(image: np.ndarray, num_chars: int = NUM_CHARS) -> List[Tuple[int, int, int, int]]:
    """Strategy 3: Fallback grid estimation based on typical container layout"""
    h, w = image.shape[:2]
    
    # Assume characters are arranged vertically in the center
    # Leave some margin on sides
    margin_x = int(w * 0.2)  # 20% margin on each side
    margin_y = int(h * 0.1)  # 10% margin top/bottom
    
    char_width = (w - 2 * margin_x) // 2  # Assume 2 columns of characters
    char_height = (h - 2 * margin_y) // (num_chars // 2 + 1)  # Distribute vertically
    
    boxes = []
    
    # Create a grid pattern
    for i in range(num_chars):
        if i < 6:  # First 6 characters in left column
            x = margin_x
            y = margin_y + i * char_height
        else:  # Remaining characters in right column
            x = margin_x + char_width
            y = margin_y + (i - 6) * char_height
            
        boxes.append((x, y, x + char_width, y + char_height))
    
    return boxes


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
    """Check if two boxes overlap significantly"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)
    
    if x2_int <= x1_int or y2_int <= y1_int:
        return False
    
    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Check if intersection is significant relative to smaller box
    min_area = min(area1, area2)
    if min_area == 0:
        return False
        
    overlap_ratio = intersection / min_area
    return overlap_ratio > threshold


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that overlap significantly"""
    if not boxes:
        return []
    
    merged = []
    for box in boxes:
        merged_with_existing = False
        for i, existing in enumerate(merged):
            if boxes_overlap(box, existing, threshold=0.3):
                # Merge boxes by taking the bounding box of both
                x1 = min(box[0], existing[0])
                y1 = min(box[1], existing[1])
                x2 = max(box[2], existing[2])
                y2 = max(box[3], existing[3])
                merged[i] = (x1, y1, x2, y2)
                merged_with_existing = True
                break
        
        if not merged_with_existing:
            merged.append(box)
    
    return merged


def filter_by_vertical_alignment(boxes: List[Tuple[int, int, int, int]], max_boxes: int = NUM_CHARS) -> List[Tuple[int, int, int, int]]:
    """Keep boxes that are most vertically aligned"""
    if len(boxes) <= max_boxes:
        return boxes
    
    # Calculate x-centers
    x_centers = [(box[0] + box[2]) / 2 for box in boxes]
    median_x = np.median(x_centers)
    
    # Calculate distance from median
    distances = [abs(x - median_x) for x in x_centers]
    
    # Get indices of boxes closest to median
    sorted_indices = np.argsort(distances)[:max_boxes]
    
    return [boxes[i] for i in sorted_indices]


def sort_boxes_spatially(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Sort boxes from top to bottom, left to right"""
    if not boxes:
        return boxes
    
    # Sort primarily by y-coordinate (top to bottom)
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def xyxy_to_yolo(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height) normalized"""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------

def process_image_hybrid(
    image_path: Path,
    model: Optional[YOLO],
    output_dir: Path,
    save_visualization: bool = False
) -> dict:
    """Process image using hybrid detection strategies"""
    
    image_name = image_path.name
    stem = image_path.stem
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return {"status": "error", "message": "Failed to read image"}
    
    h, w = image.shape[:2]
    
    # Try different detection strategies
    all_boxes = []
    strategy_results = {}
    
    # Strategy 1: YOLO detection
    if model is not None:
        yolo_boxes = yolo_detection(image, model)
        all_boxes.extend(yolo_boxes)
        strategy_results["yolo"] = len(yolo_boxes)
        logger.debug(f"YOLO found {len(yolo_boxes)} boxes")
    
    # Strategy 2: Contour detection
    contour_boxes = contour_detection(image)
    all_boxes.extend(contour_boxes)
    strategy_results["contour"] = len(contour_boxes)
    logger.debug(f"Contour found {len(contour_boxes)} boxes")
    
    # If we have very few detections, add grid estimation
    if len(all_boxes) < NUM_CHARS // 2:
        grid_boxes = grid_estimation(image)
        all_boxes.extend(grid_boxes)
        strategy_results["grid"] = len(grid_boxes)
        logger.debug(f"Grid estimation added {len(grid_boxes)} boxes")
    
    # Process all detected boxes
    if all_boxes:
        # Merge overlapping boxes
        merged_boxes = merge_overlapping_boxes(all_boxes)
        
        # Filter by vertical alignment
        filtered_boxes = filter_by_vertical_alignment(merged_boxes)
        
        # Sort spatially
        final_boxes = sort_boxes_spatially(filtered_boxes)
    else:
        # Complete fallback to grid
        final_boxes = grid_estimation(image)
        strategy_results["fallback_grid"] = len(final_boxes)
    
    # Convert to YOLO format and save
    yolo_lines = []
    for box in final_boxes:
        x_center, y_center, width, height = xyxy_to_yolo(box, w, h)
        # Use class 0 for all detections (we'll get real classes from filename later)
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Save label file
    label_path = output_dir / f"{stem}.txt"
    label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
    
    # Save visualization if requested
    if save_visualization:
        vis_image = image.copy()
        for i, box in enumerate(final_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, str(i+1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{stem}_detected.jpg"), vis_image)
    
    result = {
        "status": "success",
        "image": image_name,
        "boxes_found": len(final_boxes),
        "strategies": strategy_results
    }
    
    logger.info(f"Processed {image_name}: {len(final_boxes)} boxes using strategies {strategy_results}")
    return result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Hybrid character detection for container codes")
    parser.add_argument("--input", default=INPUT_DIR, help="Directory containing images")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory to save labels")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO model (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--no-model", action="store_true", help="Skip YOLO model loading")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting hybrid character detection...")
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model if available and requested
    model = None
    if not args.no_model and args.model and os.path.exists(args.model):
        try:
            logger.info(f"Loading YOLO model from {args.model}")
            model = YOLO(args.model)
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Continuing without YOLO detection.")
    else:
        logger.info("Skipping YOLO model (will use contour + grid detection)")
    
    # Find images
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in extensions:
        images.extend(Path(args.input).glob(f"*{ext}"))
        images.extend(Path(args.input).glob(f"*{ext.upper()}"))
    
    images = sorted(images)
    
    if args.limit:
        images = images[:args.limit]
    
    logger.info(f"Found {len(images)} images to process")
    
    # Process images
    results = []
    for image_path in images:
        try:
            result = process_image_hybrid(image_path, model, output_dir, args.visualize)
            results.append(result)
        except Exception as e:
            logger.exception(f"Error processing {image_path.name}: {e}")
            results.append({"status": "error", "image": image_path.name, "message": str(e)})
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    if successful > 0:
        avg_boxes = np.mean([r["boxes_found"] for r in results if r["status"] == "success"])
        logger.info(f"\n=== Summary ===")
        logger.info(f"Processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Average boxes per image: {avg_boxes:.1f}")
        logger.info(f"Labels saved to: {output_dir}")
        
        if args.visualize:
            logger.info(f"Visualizations saved to: {output_dir}/visualizations")
    
    # Save detailed results
    results_file = output_dir / "detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("🎉 Hybrid detection completed!")


if __name__ == "__main__":
    main()