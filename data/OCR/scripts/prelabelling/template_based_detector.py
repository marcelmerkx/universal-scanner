"""
Template-Based Character Detection

Uses YOLO detections to learn typical character box size, then applies that template
systematically to find all 11 characters in container codes.

Strategy:
1. Get YOLO detections (any number)
2. Calculate median box size from detections
3. Use template matching or grid placement with learned size
4. Place 11 boxes using the learned template
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
CONF_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('template_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template-based Detection
# ---------------------------------------------------------------------------

def get_yolo_detections(image: np.ndarray, model: YOLO, conf_threshold: float = CONF_THRESHOLD) -> List[Tuple[int, int, int, int]]:
    """Get YOLO detections as (x1, y1, x2, y2) boxes"""
    try:
        results = model(image, conf=conf_threshold, verbose=False)[0]
        if results.boxes is None or results.boxes.data.numel() == 0:
            return []
        
        boxes = results.boxes.xyxy.cpu().numpy()
        return [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
    except Exception as e:
        logger.warning(f"YOLO detection failed: {e}")
        return []


def analyze_box_sizes(boxes: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    """Calculate median width and height from detected boxes"""
    if not boxes:
        return None, None
    
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    
    median_width = np.median(widths)
    median_height = np.median(heights)
    
    return median_width, median_height


def estimate_character_positions(image: np.ndarray, template_width: float, template_height: float, num_chars: int = NUM_CHARS) -> List[Tuple[int, int, int, int]]:
    """Estimate character positions using template size"""
    h, w = image.shape[:2]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Strategy 1: Find the main text region using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(template_width), int(template_height)))
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to find text regions
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of potential text regions
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (main text area)
        largest_contour = max(contours, key=cv2.contourArea)
        text_x, text_y, text_w, text_h = cv2.boundingRect(largest_contour)
    else:
        # Fallback: assume text is in center area
        margin = 0.2
        text_x = int(w * margin)
        text_y = int(h * margin)
        text_w = int(w * (1 - 2 * margin))
        text_h = int(h * (1 - 2 * margin))
    
    # Now place characters within the text region
    boxes = []
    
    # Determine layout: vertical or horizontal
    if text_h > text_w:
        # Vertical layout (stack characters vertically)
        char_spacing = text_h / num_chars
        for i in range(num_chars):
            y = text_y + i * char_spacing
            x = text_x + (text_w - template_width) / 2  # Center horizontally
            
            boxes.append((
                int(x),
                int(y),
                int(x + template_width),
                int(y + template_height)
            ))
    else:
        # Horizontal layout (or grid layout)
        if num_chars <= 6:
            # Single row
            char_spacing = text_w / num_chars
            for i in range(num_chars):
                x = text_x + i * char_spacing
                y = text_y + (text_h - template_height) / 2  # Center vertically
                
                boxes.append((
                    int(x),
                    int(y),
                    int(x + template_width),
                    int(y + template_height)
                ))
        else:
            # Two-row layout (common for container codes)
            chars_per_row = (num_chars + 1) // 2
            row_height = text_h / 2
            char_spacing = text_w / chars_per_row
            
            for i in range(num_chars):
                if i < chars_per_row:
                    # First row
                    row = 0
                    col = i
                else:
                    # Second row
                    row = 1
                    col = i - chars_per_row
                
                x = text_x + col * char_spacing
                y = text_y + row * row_height
                
                boxes.append((
                    int(x),
                    int(y),
                    int(x + template_width),
                    int(y + template_height)
                ))
    
    return boxes


def refine_positions_with_yolo(yolo_boxes: List[Tuple[int, int, int, int]], 
                              estimated_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Refine estimated positions using YOLO detections"""
    if not yolo_boxes:
        return estimated_boxes
    
    refined_boxes = []
    
    for est_box in estimated_boxes:
        # Find the closest YOLO detection
        est_center_x = (est_box[0] + est_box[2]) / 2
        est_center_y = (est_box[1] + est_box[3]) / 2
        
        best_yolo_box = None
        min_distance = float('inf')
        
        for yolo_box in yolo_boxes:
            yolo_center_x = (yolo_box[0] + yolo_box[2]) / 2
            yolo_center_y = (yolo_box[1] + yolo_box[3]) / 2
            
            distance = np.sqrt((est_center_x - yolo_center_x)**2 + (est_center_y - yolo_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_yolo_box = yolo_box
        
        # If YOLO detection is close enough, use it; otherwise use estimation
        if best_yolo_box and min_distance < 50:  # 50 pixel threshold
            refined_boxes.append(best_yolo_box)
        else:
            refined_boxes.append(est_box)
    
    return refined_boxes


def xyxy_to_yolo(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to YOLO format"""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def process_image_template_based(
    image_path: Path,
    model: Optional[YOLO],
    output_dir: Path,
    save_visualization: bool = False,
    global_template_size: Tuple[float, float] = None
) -> dict:
    """Process image using template-based detection"""
    
    image_name = image_path.name
    stem = image_path.stem
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return {"status": "error", "message": "Failed to read image"}
    
    h, w = image.shape[:2]
    
    # Get YOLO detections
    yolo_boxes = []
    if model is not None:
        yolo_boxes = get_yolo_detections(image, model)
    
    # Determine template size
    if global_template_size:
        template_width, template_height = global_template_size
        logger.debug(f"Using global template size: {template_width:.1f} x {template_height:.1f}")
    else:
        template_width, template_height = analyze_box_sizes(yolo_boxes)
        
        if template_width is None:
            # Fallback template size (estimated)
            template_width = w * 0.08  # 8% of image width
            template_height = h * 0.08  # 8% of image height
            logger.debug(f"Using fallback template size: {template_width:.1f} x {template_height:.1f}")
        else:
            logger.debug(f"Learned template size from {len(yolo_boxes)} YOLO boxes: {template_width:.1f} x {template_height:.1f}")
    
    # Generate character positions using template
    estimated_boxes = estimate_character_positions(image, template_width, template_height)
    
    # Refine positions with YOLO detections
    final_boxes = refine_positions_with_yolo(yolo_boxes, estimated_boxes)
    
    # Convert to YOLO format and save
    yolo_lines = []
    for box in final_boxes:
        x_center, y_center, width, height = xyxy_to_yolo(box, w, h)
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Save label file
    label_path = output_dir / f"{stem}.txt"
    label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
    
    # Save visualization if requested
    if save_visualization:
        vis_image = image.copy()
        
        # Draw YOLO detections in blue
        for box in yolo_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw final boxes in green
        for i, box in enumerate(final_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, str(i+1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{stem}_template.jpg"), vis_image)
    
    result = {
        "status": "success",
        "image": image_name,
        "yolo_detections": len(yolo_boxes),
        "final_boxes": len(final_boxes),
        "template_size": [template_width, template_height]
    }
    
    logger.info(f"Processed {image_name}: {len(yolo_boxes)} YOLO → {len(final_boxes)} final boxes")
    return result


def calculate_global_template_size(images: List[Path], model: YOLO, sample_size: int = 50) -> Tuple[float, float]:
    """Calculate global template size from a sample of images"""
    logger.info(f"Calculating global template size from {min(sample_size, len(images))} images...")
    
    all_widths = []
    all_heights = []
    
    sample_images = images[:sample_size] if sample_size < len(images) else images
    
    for image_path in sample_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        yolo_boxes = get_yolo_detections(image, model)
        if yolo_boxes:
            widths = [box[2] - box[0] for box in yolo_boxes]
            heights = [box[3] - box[1] for box in yolo_boxes]
            all_widths.extend(widths)
            all_heights.extend(heights)
    
    if all_widths and all_heights:
        global_width = np.median(all_widths)
        global_height = np.median(all_heights)
        logger.info(f"Global template size: {global_width:.1f} x {global_height:.1f} (from {len(all_widths)} detections)")
        return global_width, global_height
    else:
        logger.warning("No YOLO detections found in sample, using fallback size")
        return None, None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Template-based character detection using YOLO box sizes")
    parser.add_argument("--input", default=INPUT_DIR, help="Directory containing images")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory to save labels")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--global-template", action="store_true", help="Calculate global template size from sample")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size for global template calculation")
    parser.add_argument("--confidence", type=float, default=CONF_THRESHOLD, help="YOLO confidence threshold")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting template-based character detection...")
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        return
    
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading YOLO model from {args.model}")
    model = YOLO(args.model)
    
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
    
    # Calculate global template size if requested
    global_template_size = None
    if args.global_template:
        global_template_size = calculate_global_template_size(images, model, args.sample_size)
    
    # Process images
    results = []
    for image_path in images:
        try:
            result = process_image_template_based(
                image_path, 
                model, 
                output_dir, 
                args.visualize,
                global_template_size
            )
            results.append(result)
        except Exception as e:
            logger.exception(f"Error processing {image_path.name}: {e}")
            results.append({"status": "error", "image": image_path.name, "message": str(e)})
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    if successful > 0:
        avg_yolo = np.mean([r["yolo_detections"] for r in results if r["status"] == "success"])
        logger.info(f"\n=== Summary ===")
        logger.info(f"Processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Average YOLO detections per image: {avg_yolo:.1f}")
        logger.info(f"All images now have {NUM_CHARS} template-based boxes")
        logger.info(f"Labels saved to: {output_dir}")
        
        if args.visualize:
            logger.info(f"Visualizations saved to: {output_dir}/visualizations")
    
    # Save detailed results
    results_file = output_dir / "template_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("🎉 Template-based detection completed!")


if __name__ == "__main__":
    main()