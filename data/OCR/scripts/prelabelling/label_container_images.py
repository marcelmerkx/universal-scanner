"""
Script to generate YOLO labels for container images using character detection + filename classes.

This script takes container images, runs the OCR model to detect character regions,
filters down to the 11 boxes that most likely correspond to the 11 characters
of an ISO-6346 ocean container number, and writes YOLO label files containing the
actual character classes derived from the image file name.

Usage:
    python label_container_images.py [--limit N]
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

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

# YOLO / filtering constants
CONF_THRESHOLD: float = 0.1  # Minimum confidence for detections to be considered
IOU_THRESHOLD: float = 0.1   # IoU threshold when pruning overlapping boxes
NUM_CHARS: int = 11          # ISO container codes have exactly 11 characters

# Character mapping for container codes (A-Z, 0-9)
CONTAINER_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CLASSES_MAP = {char: idx for idx, char in enumerate(CONTAINER_CHARS)}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('container_labeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Return IoU of two boxes given as (x1, y1, x2, y2) arrays."""
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0

    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def prune_overlapping_boxes(boxes: np.ndarray, iou_threshold: float = None) -> np.ndarray:
    """Remove boxes that significantly overlap with a larger box (IoU > iou_threshold)."""
    if iou_threshold is None:
        iou_threshold = IOU_THRESHOLD
    keep_indices: List[int] = []
    for i, box in enumerate(boxes):
        should_keep = True
        area_i = (box[2] - box[0]) * (box[3] - box[1])
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if calculate_iou(box, other) > iou_threshold:
                area_j = (other[2] - other[0]) * (other[3] - other[1])
                if area_i < area_j:
                    should_keep = False
                    break
        if should_keep:
            keep_indices.append(i)
    return boxes[keep_indices]


def keep_vertically_aligned(boxes: np.ndarray, k: int = NUM_CHARS) -> np.ndarray:
    """Return k boxes whose centres are closest to the median x-position (vertical line)."""
    if len(boxes) <= k:
        return boxes

    # Compute x centre for each box
    x_centres = (boxes[:, 0] + boxes[:, 2]) / 2.0
    median_x = float(np.median(x_centres))
    deviations = np.abs(x_centres - median_x)
    closest_indices = np.argsort(deviations)[:k]
    return boxes[closest_indices]


def xyxy_to_yolo(box: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) pixel box → (x_c, y_c, w, h) all normalised 0-1."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h


def get_image_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """Get all image files from directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    image_files = []
    input_path = Path(input_dir)
    
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def process_container_image(
    img_path: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    allow_partial: bool = False,
    min_detections: int = NUM_CHARS,
) -> bool:
    """Process container image and write label file to output directory.

    Returns True on success, False when the image is skipped (e.g. < 11 detections).
    """
    filename = img_path.name
    stem = img_path.stem.upper()
    
    if len(stem) < NUM_CHARS:
        logger.warning("Skipping %s: file name has fewer than %d characters", filename, NUM_CHARS)
        return False
    
    target_chars = stem[:NUM_CHARS]
    
    # Validate characters are in our character set
    for char in target_chars:
        if char not in CLASSES_MAP:
            logger.warning("Skipping %s: invalid character '%s' in filename", filename, char)
            return False

    # Load image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        logger.error("Failed to read image: %s", img_path)
        return False
    img_h, img_w = img_bgr.shape[:2]

    # Run YOLO inference
    results = model(img_bgr, conf=conf_threshold, verbose=False)[0]
    if results.boxes is None or results.boxes.data.numel() == 0:
        logger.warning("No detections in %s", filename)
        return False

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # shape (N, 4)
    initial_count = len(boxes_xyxy)

    # Remove overlapping boxes and keep vertically aligned
    boxes_xyxy = prune_overlapping_boxes(boxes_xyxy, iou_threshold)
    after_nms_count = len(boxes_xyxy)
    
    boxes_xyxy = keep_vertically_aligned(boxes_xyxy)
    after_alignment_count = len(boxes_xyxy)
    
    logger.info("Detection stages for %s: initial=%d, after_nms=%d, after_alignment=%d", 
                filename, initial_count, after_nms_count, after_alignment_count)

    detected_count = len(boxes_xyxy)
    
    if not allow_partial and detected_count != NUM_CHARS:
        logger.warning("Skipping %s: expected %d detections, got %d", filename, NUM_CHARS, detected_count)
        return False
    
    if detected_count < min_detections:
        logger.warning("Skipping %s: minimum %d detections required, got %d", filename, min_detections, detected_count)
        return False

    # Order top-to-bottom (ascending y1)
    sorted_indices = np.argsort(boxes_xyxy[:, 1])
    boxes_xyxy = boxes_xyxy[sorted_indices]

    # Prepare label lines
    label_lines: List[str] = []
    chars_to_use = min(detected_count, len(target_chars))
    
    for idx, box in enumerate(boxes_xyxy[:chars_to_use]):
        char = target_chars[idx]
        cls_id = CLASSES_MAP[char]
        x_c, y_c, w, h = xyxy_to_yolo(box, img_w, img_h)
        label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Write label file in output directory
    label_path = output_dir / f"{img_path.stem}.txt"
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    logger.info("Processed %-20s → %s chars (%d/%d)", filename, target_chars[:chars_to_use], chars_to_use, NUM_CHARS)
    return True


def main():
    """Main function to label container images."""
    parser = argparse.ArgumentParser(description="Label container images using filename-based character classes")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained YOLOv8 model (.pt file)")
    parser.add_argument("--input", default=INPUT_DIR, help="Directory containing images to label")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory to save YOLO format annotations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--confidence", type=float, default=CONF_THRESHOLD, help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD, help=f"IoU threshold for overlap removal (default: {IOU_THRESHOLD})")
    parser.add_argument("--allow-partial", action="store_true", help="Allow processing images with fewer than 11 detections")
    parser.add_argument("--min-detections", type=int, default=NUM_CHARS, help=f"Minimum detections required (default: {NUM_CHARS})")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting container images labeling process...")
    
    # Validate inputs
    if not args.model:
        logger.error("Model path not specified. Use --model or set MODEL_PATH in config_template.py")
        return
    
    if not args.input:
        logger.error("Input directory not specified. Use --input or set INPUT_DIR in config_template.py")
        return
    
    if not args.output:
        logger.error("Output directory not specified. Use --output or set OUTPUT_DIR in config_template.py")
        return

    if not os.path.exists(args.model):
        logger.error("Model file not found: %s", args.model)
        return

    if not os.path.exists(args.input):
        logger.error("Input directory not found: %s", args.input)
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use local variables for thresholds
    conf_threshold = args.confidence
    iou_threshold = args.iou_threshold
    
    logger.info("Using confidence threshold: %.3f", conf_threshold)
    logger.info("Using IoU threshold: %.3f", iou_threshold)
    
    # Load YOLO model
    logger.info("Loading YOLO model from %s…", args.model)
    model = YOLO(str(args.model))
    
    # Find all images
    images = get_image_files(args.input)
    
    if not images:
        logger.warning("No images found in %s", args.input)
        return
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        images = images[:args.limit]
        logger.info("Limiting processing to %d images", len(images))
    
    logger.info("Found %d image(s) to process", len(images))
    
    # Process each image
    processed = 0
    skipped = 0
    
    for img_path in images:
        try:
            if process_container_image(img_path, model, output_dir, conf_threshold, iou_threshold, args.allow_partial, args.min_detections):
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.exception("Error processing %s: %s", img_path.name, exc)
            skipped += 1
    
    logger.info("\n=== Summary ===")
    logger.info("Processed: %d", processed)
    logger.info("Skipped:   %d", skipped)
    
    if processed > 0:
        logger.info("🎉 Labels saved in %s", output_dir)
        logger.info("Each processed image now has a corresponding .txt label file.")


if __name__ == "__main__":
    main()