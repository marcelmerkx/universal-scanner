#!/usr/bin/env python3
"""Generate YOLOv8 label files using OCR model predictions with ISO 6346 validation.

This script processes images from input folder subfolders (test, val, train),
uses a YOLOv8 OCR model to detect and recognize characters, validates the
resulting container codes against ISO 6346 checksum, and generates YOLO
label files for valid codes.

Steps per input image:
1. Load image from input folder subfolders
2. Run OCR model to detect character regions and classify them
3. Order detections top-to-bottom and assemble container code
4. Apply container code corrections (common OCR mistakes)
5. Validate against ISO 6346 checksum
6. If valid, copy image and create label file; if invalid, log for review

Run:
    python3 scripts/generate_ocr_dataset.py --limit 10  # Dry-run on 10 files
    python3 scripts/generate_ocr_dataset.py             # Full dataset
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO  # type: ignore

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("notebooks/vertical-container-areas-public-cc-license-1")
OUT_IMAGES_DIR = Path("data/dataset/images/cc")
OUT_LABELS_DIR = Path("data/dataset/labels/cc")
MODEL_PATH = Path("models/best-OCR-18-06-25.pt")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_THRESHOLD: float = 0.1
IOU_THRESHOLD: float = 0.3
NUM_CHARS: int = 11
SUBFOLDER_NAMES = ["test", "valid", "train"]
FAILED_SUBFOLDER = "failed"

# ---------------------------------------------------------------------------
# ISO 6346 validation
# ---------------------------------------------------------------------------

# ISO 6346 letter to numeric value mapping
LETTER_VALUES = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18,
    "I": 19, "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27,
    "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36,
    "Y": 37, "Z": 38
}

def char_to_value(ch: str) -> Optional[int]:
    """Convert ISO 6346 character to numeric value."""
    if ch.isdigit():
        return int(ch)
    return LETTER_VALUES.get(ch.upper())

def compute_iso6346_check_digit(code_without_check: str) -> Optional[int]:
    """Compute ISO 6346 check digit for first 10 characters."""
    if len(code_without_check) != 10:
        return None
    
    total = 0
    for i, ch in enumerate(code_without_check):
        val = char_to_value(ch)
        if val is None:
            return None
        weight = 2 ** i
        total += val * weight
    
    remainder = total % 11
    return 0 if remainder == 10 else remainder

def is_valid_iso6346_code(full_code: str) -> bool:
    """Validate full 11-character ISO 6346 container code."""
    if len(full_code) != 11:
        return False
    
    upper_code = full_code.upper()
    code_part = upper_code[:10]
    check_digit = upper_code[10]
    
    # Basic format validation
    if not code_part[:4].isalpha() or not code_part[4:].isdigit() or not check_digit.isdigit():
        return False
    
    expected_digit = compute_iso6346_check_digit(code_part)
    if expected_digit is None:
        return False
    
    return int(check_digit) == expected_digit

# ---------------------------------------------------------------------------
# Container code corrections
# ---------------------------------------------------------------------------

def apply_container_code_corrections(text: str) -> str:
    """Apply common OCR corrections for container codes."""
    if not text:
        return text
    
    # Clean up
    cleaned = text.replace(" ", "").replace("-", "").upper()
    
    # First 4 characters should be letters
    if len(cleaned) >= 4:
        prefix = cleaned[:4]
        corrected_prefix = (prefix
                          .replace("0", "O")
                          .replace("4", "A") 
                          .replace("8", "B")
                          .replace("5", "S")
                          .replace("1", "I"))
        cleaned = corrected_prefix + cleaned[4:]
    
    # Characters 5-11 should be digits
    if len(cleaned) >= 5:
        suffix = cleaned[4:]
        corrected_suffix = (suffix
                          .replace("O", "0")
                          .replace("A", "4")
                          .replace("B", "8") 
                          .replace("S", "5")
                          .replace("I", "1"))
        cleaned = cleaned[:4] + corrected_suffix
    
    return cleaned

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_classes_from_model(model: YOLO) -> Dict[str, int]:
    """Load character to class index mapping from YOLO model."""
    classes: Dict[str, int] = {}
    for idx, name in model.names.items():
        char = name.strip().upper()
        if len(char) != 1:
            raise ValueError(f"Invalid class name '{char}' (must be single character)")
        classes[char] = idx

    if len(classes) != 36:
        logger.warning(
            "Expected 36 classes (A-Z, 0-9) but found %d from model", len(classes)
        )
    return classes

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in xyxy format."""
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

def prune_overlapping_boxes(boxes: List[List[float]], classes: List[int], iou_threshold: float = IOU_THRESHOLD) -> Tuple[List[List[float]], List[int]]:
    """Remove overlapping boxes, keeping larger ones."""
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
    return [boxes[i] for i in keep_indices], [classes[i] for i in keep_indices]

def keep_vertically_aligned(boxes: List[List[float]], classes: List[int], k: int = NUM_CHARS) -> Tuple[List[List[float]], List[int]]:
    """Keep k boxes closest to median x-position (vertical alignment)."""
    if len(boxes) <= k:
        return boxes, classes

    # Calculate x centers
    x_centres = [(box[0] + box[2]) / 2.0 for box in boxes]
    
    # Calculate median manually
    sorted_x = sorted(x_centres)
    n = len(sorted_x)
    median_x = sorted_x[n // 2] if n % 2 == 1 else (sorted_x[n // 2 - 1] + sorted_x[n // 2]) / 2.0
    
    # Calculate deviations and get indices sorted by deviation
    deviations = [abs(x - median_x) for x in x_centres]
    closest_indices = sorted(range(len(deviations)), key=lambda i: deviations[i])[:k]
    
    return [boxes[i] for i in closest_indices], [classes[i] for i in closest_indices]

def xyxy_to_yolo(box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert xyxy pixel box to YOLO normalized format."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h

def process_image(
    img_path: Path,
    model: YOLO,
    classes_map: Dict[str, int],
    dst_imgs_dir: Path,
    dst_lbls_dir: Path,
    subfolder: str,
    invalid_codes_log: List[str],
) -> bool:
    """Process single image with OCR model and validate container code."""
    filename = img_path.name
    
    # Run YOLO OCR inference directly on image path
    results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]
    if results.boxes is None or results.boxes.data.numel() == 0:
        logger.warning("No detections in %s", filename)
        return False

    # Get image dimensions from YOLO results
    img_h, img_w = results.orig_shape

    # Convert tensors to Python lists
    boxes_xyxy = results.boxes.xyxy.cpu().tolist()
    classes_tensor = [int(cls) for cls in results.boxes.cls.cpu().tolist()]

    # Remove overlapping boxes and keep vertically aligned
    boxes_xyxy, classes_tensor = prune_overlapping_boxes(boxes_xyxy, classes_tensor)
    boxes_xyxy, classes_tensor = keep_vertically_aligned(boxes_xyxy, classes_tensor)

    if len(boxes_xyxy) != NUM_CHARS:
        logger.warning("Skipping %s: expected %d detections, got %d", filename, NUM_CHARS, len(boxes_xyxy))
        return False

    # Order top-to-bottom (ascending y1) - sort by y1 coordinate
    sorted_data = sorted(zip(boxes_xyxy, classes_tensor), key=lambda x: x[0][1])
    boxes_xyxy, classes_tensor = zip(*sorted_data)
    boxes_xyxy, classes_tensor = list(boxes_xyxy), list(classes_tensor)

    # Convert class indices to characters
    reverse_classes_map = {v: k for k, v in classes_map.items()}
    detected_chars = []
    for cls_idx in classes_tensor:
        if cls_idx in reverse_classes_map:
            detected_chars.append(reverse_classes_map[cls_idx])
        else:
            logger.error("Unknown class index %d in %s", cls_idx, filename)
            return False

    # Assemble container code
    raw_code = "".join(detected_chars)
    corrected_code = apply_container_code_corrections(raw_code)
    
    # Create label lines using corrected characters (needed for both valid and failed cases)
    label_lines: List[str] = []
    for idx, box in enumerate(boxes_xyxy):
        char = corrected_code[idx]
        if char not in classes_map:
            logger.error("Corrected character '%s' not in classes for %s", char, filename)
            return False
        cls_id = classes_map[char]
        x_c, y_c, w, h = xyxy_to_yolo(box, img_w, img_h)
        label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    # Validate ISO 6346 checksum
    is_valid = is_valid_iso6346_code(corrected_code)
    
    if not is_valid:
        invalid_entry = f"{subfolder}/{filename}: raw='{raw_code}' corrected='{corrected_code}' - INVALID CHECKSUM"
        invalid_codes_log.append(invalid_entry)
        logger.warning("Invalid ISO 6346 code in %s: %s", filename, invalid_entry)
        
        # Save failed case to failed folder
        failed_img_dir = dst_imgs_dir / FAILED_SUBFOLDER
        failed_lbl_dir = dst_lbls_dir / FAILED_SUBFOLDER
        failed_img_dir.mkdir(parents=True, exist_ok=True)
        failed_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Use corrected (but invalid) code as filename for failed cases
        file_extension = img_path.suffix.lower()
        failed_filename = f"{corrected_code}{file_extension}"
        failed_img_path = failed_img_dir / failed_filename
        failed_lbl_path = failed_lbl_dir / f"{corrected_code}.txt"
        
        # Check for collisions in failed folder too
        if failed_img_path.exists():
            logger.info("Skipping failed %-20s → %s (duplicate in failed folder)", f"{subfolder}/{filename}", corrected_code)
            return False
        
        # Copy image and write label file to failed folder
        shutil.copy2(img_path, failed_img_path)
        failed_lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        
        logger.info("Saved failed %-20s → %s (invalid checksum)", f"{subfolder}/{filename}", corrected_code)
        return False

    # Ensure output directories exist
    subfolder_img_dir = dst_imgs_dir / subfolder
    subfolder_lbl_dir = dst_lbls_dir / subfolder
    subfolder_img_dir.mkdir(parents=True, exist_ok=True)
    subfolder_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Use container code as filename
    file_extension = img_path.suffix.lower()
    new_filename = f"{corrected_code}{file_extension}"
    new_img_path = subfolder_img_dir / new_filename
    new_lbl_path = subfolder_lbl_dir / f"{corrected_code}.txt"
    
    # Check for collisions (same container code already exists)
    if new_img_path.exists():
        logger.info("Skipping %-20s → %s (duplicate container code)", f"{subfolder}/{filename}", corrected_code)
        return False
    
    # Copy image and write label file with new names
    shutil.copy2(img_path, new_img_path)
    new_lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    logger.info("Processed %-20s → %s (valid ISO 6346)", f"{subfolder}/{filename}", corrected_code)
    return True

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset using OCR model with ISO 6346 validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Input folder with subfolders")
    parser.add_argument("--output-images", type=Path, default=OUT_IMAGES_DIR, help="Output images folder")
    parser.add_argument("--output-labels", type=Path, default=OUT_LABELS_DIR, help="Output labels folder")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="OCR YOLO model path (.pt)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images per subfolder (0 = no limit)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Detection confidence threshold")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    global CONF_THRESHOLD
    CONF_THRESHOLD = args.conf

    # Load model
    logger.info("Loading OCR YOLO model from %s…", args.model)
    model = YOLO(str(args.model))
    
    # Load classes from model
    classes_map = load_classes_from_model(model)
    logger.info("Loaded %d classes from model", len(classes_map))

    # Track statistics
    total_processed = 0
    total_skipped = 0
    invalid_codes_log: List[str] = []

    # Process each subfolder
    for subfolder in SUBFOLDER_NAMES:
        subfolder_path = args.input_dir / subfolder
        if not subfolder_path.is_dir():
            logger.warning("Subfolder not found: %s", subfolder_path)
            continue

        # Find images in subfolder/images/
        images_path = subfolder_path / "images"
        if not images_path.is_dir():
            logger.warning("Images directory not found: %s", images_path)
            continue
            
        images = sorted(p for p in images_path.iterdir() 
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        
        if args.limit > 0:
            images = images[:args.limit]
        
        logger.info("Processing %s: found %d image(s)", subfolder, len(images))

        subfolder_processed = 0
        subfolder_skipped = 0

        for img_path in images:
            try:
                if process_image(img_path, model, classes_map, args.output_images, 
                               args.output_labels, subfolder, invalid_codes_log):
                    subfolder_processed += 1
                else:
                    subfolder_skipped += 1
            except Exception as exc:
                logger.exception("Error processing %s: %s", img_path.name, exc)
                subfolder_skipped += 1

        logger.info("%s summary: processed=%d, skipped=%d", subfolder, subfolder_processed, subfolder_skipped)
        total_processed += subfolder_processed
        total_skipped += subfolder_skipped

    # Write invalid codes log
    if invalid_codes_log:
        invalid_log_path = args.output_labels.parent / "invalid_codes.log"
        invalid_log_path.write_text("\n".join(invalid_codes_log) + "\n", encoding="utf-8")
        logger.info("Wrote %d invalid codes to %s", len(invalid_codes_log), invalid_log_path)

    logger.info("\n=== Final Summary ===")
    logger.info("Total processed: %d", total_processed)
    logger.info("Total skipped:   %d", total_skipped)
    logger.info("Invalid codes:   %d", len(invalid_codes_log))

if __name__ == "__main__":
    main() 