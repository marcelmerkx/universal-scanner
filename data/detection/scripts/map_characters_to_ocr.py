#!/usr/bin/env python3
"""
Map character detections to OCR labels based on filename.

This script:
1. Finds images with exactly 11 character detections
2. Maps detections from left to right to the first 11 characters of the filename
3. Converts class 0 (character) to OCR classes (0-9, A-Z)
4. Copies images and labels to 03_ocr_generated folder

Usage:
    python3 data/detection/scripts/map_characters_to_ocr.py
    python3 data/detection/scripts/map_characters_to_ocr.py --limit 10
    python3 data/detection/scripts/map_characters_to_ocr.py --dry-run
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import numpy as np

# Configuration
CHARACTERS_DIR = Path("data/detection/training_data/02_characters")
OCR_OUTPUT_DIR = Path("data/detection/training_data/03_ocr_generated")
LOG_DIR = Path("data/detection/logs")

# OCR class mapping from best-OCR-18-06-25-data.yaml
# Class indices: 0-9 for digits, 10-35 for A-Z
OCR_CLASS_MAP = {
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
    log_file = LOG_DIR / "map_characters_to_ocr.log"
    
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
    Extract the first 11 characters from filename (container code).
    
    Examples:
        "SUDU1234567-1.jpg" -> "SUDU1234567"
        "HASU4811682-1.jpg" -> "HASU4811682"
    """
    # Remove extension and any suffix after the container code
    base_name = Path(filename).stem
    
    # Container codes are typically 11 characters (4 letters + 7 digits)
    # Extract the first 11 alphanumeric characters
    match = re.match(r'^([A-Z]{4}\d{7})', base_name)
    if match:
        return match.group(1)
    
    # Fallback: just take first 11 characters
    return base_name[:11] if len(base_name) >= 11 else base_name


def calculate_overlap(det1: Tuple[int, float, float, float, float], 
                     det2: Tuple[int, float, float, float, float]) -> float:
    """
    Calculate overlap percentage between two detections.
    
    Returns:
        Overlap as percentage (0.0 to 1.0)
    """
    # Extract bounding box coordinates (convert from YOLO format)
    # det format: (class_id, center_x, center_y, width, height)
    
    # Detection 1
    cx1, cy1, w1, h1 = det1[1], det1[2], det1[3], det1[4]
    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    area1 = w1 * h1
    
    # Detection 2
    cx2, cy2, w2, h2 = det2[1], det2[2], det2[3], det2[4]
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2
    area2 = w2 * h2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate overlap percentage relative to smaller box
    smaller_area = min(area1, area2)
    if smaller_area == 0:
        return 0.0
    
    return intersection / smaller_area


def remove_overlapping_detections(detections: List[Tuple[int, float, float, float, float]], 
                                 overlap_threshold: float = 0.5) -> List[Tuple[int, float, float, float, float]]:
    """
    Remove overlapping detections, keeping the larger box.
    
    Args:
        detections: List of detections (class_id, center_x, center_y, width, height)
        overlap_threshold: Overlap threshold (0.5 = 50%)
        
    Returns:
        Filtered list of detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by area (largest first) to prioritize keeping larger boxes
    detections_with_area = []
    for det in detections:
        area = det[3] * det[4]  # width * height
        detections_with_area.append((det, area))
    
    detections_with_area.sort(key=lambda x: x[1], reverse=True)  # Sort by area, largest first
    
    filtered = []
    
    for det, area in detections_with_area:
        # Check if this detection overlaps significantly with any already kept detection
        keep_detection = True
        
        for kept_det in filtered:
            overlap = calculate_overlap(det, kept_det)
            if overlap > overlap_threshold:
                keep_detection = False
                break
        
        if keep_detection:
            filtered.append(det)
    
    return filtered


def find_character_sequence(detections: List[Tuple[int, float, float, float, float]], 
                           target_length: int = 11) -> List[Tuple[int, float, float, float, float]]:
    """
    Find the best sequence of characters that likely represents the container code.
    Uses RANSAC-like approach to find the main text line.
    
    Args:
        detections: List of detections (class_id, center_x, center_y, width, height)
        target_length: Expected sequence length (11 for container codes)
        
    Returns:
        Best sequence of detections forming a continuous path
    """
    # Always apply line-finding logic, even if we have exactly target_length detections
    # This ensures we pick the best 11 detections that form a line, not just any 11
    
    # Use RANSAC-like approach to find the main line
    best_sequence = []
    best_score = 0
    
    # Try different pairs of detections to define a line
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det1 = detections[i]
            det2 = detections[j]
            
            # Skip if detections are too close in X
            if abs(det1[1] - det2[1]) < 0.05:
                continue
            
            # Calculate line parameters (y = mx + b)
            x1, y1 = det1[1], det1[2]
            x2, y2 = det2[1], det2[2]
            
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Find all detections close to this line
                line_detections = []
                avg_height = (det1[4] + det2[4]) / 2
                
                for det in detections:
                    x, y = det[1], det[2]
                    expected_y = slope * x + intercept
                    
                    # Check if detection is close to the line
                    if abs(y - expected_y) < avg_height * 0.25:  # Very strict tolerance
                        line_detections.append(det)
                
                # If we found enough detections on this line
                if len(line_detections) >= target_length - 1:
                    # Sort by x coordinate
                    line_detections.sort(key=lambda d: d[1])
                    
                    # Take the first 11 detections
                    sequence = line_detections[:target_length]
                    
                    # Score this sequence
                    score = calculate_sequence_score(sequence, target_length)
                    
                    # Bonus for reaching target length
                    if len(sequence) == target_length:
                        score += 0.5
                    
                    # Bonus for low Y variance (straight line)
                    y_values = [d[2] for d in sequence]
                    expected_y_values = [slope * d[1] + intercept for d in sequence]
                    y_error = sum(abs(y - ey) for y, ey in zip(y_values, expected_y_values)) / len(sequence)
                    linearity_bonus = 1.0 / (1.0 + y_error * 10)
                    score += linearity_bonus * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_sequence = sequence
    
    # If RANSAC didn't find a good line, fall back to the original corridor approach
    if not best_sequence or len(best_sequence) < target_length:
        # Sort detections by x coordinate
        x_sorted = sorted(detections, key=lambda d: d[1])
        
        # Find the most populated horizontal band
        y_values = [d[2] for d in x_sorted]
        median_y = np.median(y_values)
        avg_height = np.mean([d[4] for d in x_sorted])
        
        # Get detections in the main horizontal band
        main_band = [d for d in x_sorted if abs(d[2] - median_y) < avg_height * 0.4]
        
        if len(main_band) >= target_length:
            best_sequence = main_band[:target_length]
        else:
            # Use corridor approach as last resort
            start_det = x_sorted[0]
            sequence = [start_det]
            remaining = [det for det in x_sorted[1:]]  
            
            current_det = start_det
            prev_det = None
            
            while len(sequence) < target_length and remaining:
                next_det = find_next_in_corridor(current_det, remaining, prev_det)
                
                if next_det is None:
                    break
                    
                sequence.append(next_det)
                remaining.remove(next_det)
                prev_det = current_det
                current_det = next_det
            
            best_sequence = sequence
    
    return sort_detections_by_x(best_sequence)


def find_next_in_corridor(current_det: Tuple[int, float, float, float, float], 
                         candidates: List[Tuple[int, float, float, float, float]],
                         prev_det: Optional[Tuple[int, float, float, float, float]] = None) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Find the next character in the corridor from current position.
    
    Args:
        current_det: Current detection (class_id, center_x, center_y, width, height)
        candidates: List of remaining candidate detections
        prev_det: Previous detection to maintain direction consistency
        
    Returns:
        Best next detection or None if no valid candidate
    """
    if not candidates:
        return None
    
    curr_x, curr_y, curr_w, curr_h = current_det[1], current_det[2], current_det[3], current_det[4]
    
    # Define corridor parameters (very strict)
    max_x_distance = curr_w * 1.8  # Maximum horizontal distance (1.8x character width) - very tight
    max_y_distance = curr_h * 0.3  # Maximum vertical distance (0.3x character height) - very tight
    min_x_distance = curr_w * 0.4  # Minimum horizontal distance to avoid overlap
    
    # If we have a previous detection, maintain direction consistency
    expected_x_step = None
    expected_y_step = None
    if prev_det is not None:
        prev_x, prev_y = prev_det[1], prev_det[2]
        expected_x_step = curr_x - prev_x
        expected_y_step = curr_y - prev_y
    
    valid_candidates = []
    
    for candidate in candidates:
        cand_x, cand_y, cand_w, cand_h = candidate[1], candidate[2], candidate[3], candidate[4]
        
        # Check if candidate is to the right and within corridor
        x_dist = cand_x - curr_x
        y_dist = abs(cand_y - curr_y)
        
        # Must be to the right, not too far, and within vertical corridor
        if (min_x_distance <= x_dist <= max_x_distance and 
            y_dist <= max_y_distance):
            
            # Score based on distance and size similarity
            distance_score = 1.0 / (1.0 + x_dist)  # Closer is better
            y_alignment_score = 1.0 / (1.0 + y_dist * 2)  # Better vertical alignment
            size_similarity = min(curr_w, cand_w) / max(curr_w, cand_w)  # Similar sizes preferred
            
            # Direction consistency bonus
            direction_bonus = 1.0
            if expected_x_step is not None and expected_y_step is not None:
                # Check if this candidate continues the same direction
                actual_x_step = cand_x - curr_x
                actual_y_step = cand_y - curr_y
                
                # Prefer candidates that maintain similar step size and direction
                x_consistency = 1.0 / (1.0 + abs(actual_x_step - expected_x_step) * 3)  # Heavily penalize direction changes
                y_consistency = 1.0 / (1.0 + abs(actual_y_step - expected_y_step) * 3)
                direction_bonus = (x_consistency + y_consistency) / 2
            
            # Increase weight on alignment and direction consistency
            total_score = distance_score * 0.2 + y_alignment_score * 0.4 + size_similarity * 0.1 + direction_bonus * 0.3
            
            valid_candidates.append((candidate, total_score))
    
    if not valid_candidates:
        return None
    
    # Return the best scored candidate
    valid_candidates.sort(key=lambda x: x[1], reverse=True)
    return valid_candidates[0][0]


def calculate_sequence_score(sequence: List[Tuple[int, float, float, float, float]], 
                           target_length: int) -> float:
    """
    Calculate a score for how good a character sequence is.
    
    Returns:
        Score (higher is better)
    """
    if not sequence:
        return 0.0
    
    # Length score (prefer sequences closer to target length)
    length_score = min(len(sequence) / target_length, 1.0)
    
    # Spacing consistency score
    if len(sequence) < 2:
        spacing_score = 1.0
    else:
        spacings = []
        for i in range(len(sequence) - 1):
            curr_x = sequence[i][1]
            next_x = sequence[i + 1][1]
            spacings.append(next_x - curr_x)
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_spacing = sum(spacings) / len(spacings)
        if mean_spacing > 0:
            spacing_variance = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
            spacing_std = spacing_variance ** 0.5
            cv = spacing_std / mean_spacing
            spacing_score = 1.0 / (1.0 + cv)  # Lower CV = higher score
        else:
            spacing_score = 0.0
    
    # Size consistency score
    if len(sequence) < 2:
        size_score = 1.0
    else:
        widths = [det[3] for det in sequence]
        heights = [det[4] for det in sequence]
        
        # Calculate consistency for both width and height
        mean_w, mean_h = sum(widths) / len(widths), sum(heights) / len(heights)
        
        if mean_w > 0 and mean_h > 0:
            w_cv = (sum((w - mean_w) ** 2 for w in widths) / len(widths)) ** 0.5 / mean_w
            h_cv = (sum((h - mean_h) ** 2 for h in heights) / len(heights)) ** 0.5 / mean_h
            size_score = 1.0 / (1.0 + (w_cv + h_cv) / 2)
        else:
            size_score = 0.0
    
    # Continuity score - penalty for large gaps
    if len(sequence) < 2:
        continuity_score = 1.0
    else:
        # Calculate the standard deviation of gaps
        gaps = []
        for i in range(len(sequence) - 1):
            curr_x = sequence[i][1]
            next_x = sequence[i + 1][1]
            gaps.append(next_x - curr_x)
        
        # Penalize sequences with very inconsistent gaps
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap > 0:
            gap_variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            gap_std = gap_variance ** 0.5
            # Penalty for large relative variation in gaps
            continuity_score = 1.0 / (1.0 + (gap_std / mean_gap))
        else:
            continuity_score = 0.0
    
    # Combined score with higher weight on length and continuity
    total_score = length_score * 0.4 + spacing_score * 0.2 + size_score * 0.2 + continuity_score * 0.2
    
    return total_score


def sort_detections_by_x(detections: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    """
    Sort detections from left to right based on center_x coordinate.
    """
    return sorted(detections, key=lambda d: d[1])  # Sort by center_x


def adjust_overlapping_boxes(detections: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    """
    Adjust bounding box widths to remove overlaps by averaging the overlapping areas.
    
    Args:
        detections: List of detections sorted by x coordinate
        
    Returns:
        List of detections with adjusted widths to prevent overlap
    """
    if len(detections) <= 1:
        return detections
    
    adjusted_detections = []
    
    for i, det in enumerate(detections):
        class_id, center_x, center_y, width, height = det
        
        # Calculate original left and right edges
        left_edge = center_x - width / 2
        right_edge = center_x + width / 2
        
        # Adjust left edge based on previous detection
        if i > 0:
            prev_det = detections[i - 1]
            prev_right_edge = prev_det[1] + prev_det[3] / 2
            
            # If there's overlap with previous box
            if prev_right_edge > left_edge:
                # Split the overlap - midpoint between the two edges
                new_boundary = (prev_right_edge + left_edge) / 2
                left_edge = new_boundary
        
        # Adjust right edge based on next detection
        if i < len(detections) - 1:
            next_det = detections[i + 1]
            next_left_edge = next_det[1] - next_det[3] / 2
            
            # If there's overlap with next box
            if right_edge > next_left_edge:
                # Split the overlap - midpoint between the two edges
                new_boundary = (right_edge + next_left_edge) / 2
                right_edge = new_boundary
        
        # Calculate new width and center
        new_width = right_edge - left_edge
        new_center_x = (left_edge + right_edge) / 2
        
        # Ensure width is positive and reasonable
        if new_width > 0:
            adjusted_detections.append((class_id, new_center_x, center_y, new_width, height))
        else:
            # Fallback to original if something went wrong
            adjusted_detections.append(det)
    
    return adjusted_detections


def map_detections_to_ocr(detections: List[Tuple[int, float, float, float, float]], 
                         container_code: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Map character detections to OCR classes based on container code.
    
    Args:
        detections: List of detections sorted left to right
        container_code: 11-character container code
        
    Returns:
        List of detections with updated class IDs
    """
    mapped_detections = []
    
    for i, (_, center_x, center_y, width, height) in enumerate(detections):
        if i < len(container_code):
            char = container_code[i].upper()
            if char in OCR_CLASS_MAP:
                ocr_class_id = OCR_CLASS_MAP[char]
                mapped_detections.append((ocr_class_id, center_x, center_y, width, height))
            else:
                # Skip unknown characters
                continue
    
    return mapped_detections


def write_yolo_label(label_path: Path, detections: List[Tuple[int, float, float, float, float]]):
    """Write detections to YOLO format label file."""
    with open(label_path, 'w') as f:
        for class_id, center_x, center_y, width, height in detections:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def copy_to_error_folder(image_path: Path, label_path: Path, error_images_dir: Path, 
                        error_labels_dir: Path, reason: str, logger: logging.Logger,
                        adjusted_detections: Optional[List[Tuple[int, float, float, float, float]]] = None):
    """Copy problematic image and label to error folders."""
    try:
        # Copy image
        error_image_path = error_images_dir / image_path.name
        shutil.copy2(image_path, error_image_path)
        
        # If we have adjusted detections, write them instead of copying original
        error_label_path = error_labels_dir / label_path.name
        if adjusted_detections:
            write_yolo_label(error_label_path, adjusted_detections)
            logger.debug(f"Wrote adjusted label with {len(adjusted_detections)} detections to error folder")
        else:
            # Copy label (original, not mapped)
            shutil.copy2(label_path, error_label_path)
        
        logger.debug(f"Copied {image_path.name} to error folder (reason: {reason})")
        
    except Exception as e:
        logger.error(f"Failed to copy {image_path.name} to error folder: {e}")


def process_images(images_dir: Path, labels_dir: Path, output_images_dir: Path, 
                  output_labels_dir: Path, limit: int, dry_run: bool, logger: logging.Logger) -> Dict:
    """
    Process images with exactly 11 detections and map to OCR labels.
    
    Returns:
        Dictionary with processing statistics
    """
    # Create output directories
    if not dry_run:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create error directories for problematic images
        error_base_dir = Path("data/detection/training_data/04_ocr_error")
        error_images_dir = error_base_dir / "images"
        error_labels_dir = error_base_dir / "labels"
        error_images_dir.mkdir(parents=True, exist_ok=True)
        error_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all label files
    label_files = sorted(list(labels_dir.glob("*.txt")))
    
    # Statistics
    stats = {
        "total_labels": len(label_files),
        "labels_processed": 0,
        "labels_with_11_detections": 0,
        "labels_mapped": 0,
        "labels_skipped": 0,
        "labels_copied_to_error": 0,
        "failed_labels": 0,
        "character_distribution": {},
        "error_reasons": {}
    }
    
    # Process each label file
    processed_count = 0
    for label_path in label_files:
        if limit and processed_count >= limit:
            break
            
        # Find corresponding image
        image_name = label_path.stem + ".jpg"
        image_path = images_dir / image_name
        
        if not image_path.exists():
            # Try other extensions
            for ext in ['.png', '.jpeg', '.bmp']:
                alt_path = images_dir / (label_path.stem + ext)
                if alt_path.exists():
                    image_path = alt_path
                    image_name = alt_path.name
                    break
            else:
                logger.warning(f"No image found for label: {label_path.name}")
                stats["failed_labels"] += 1
                continue
        
        try:
            # Parse label file
            detections = parse_yolo_label(label_path)
            stats["labels_processed"] += 1
            
            # Check if exactly 11 detections
            if len(detections) != 11:
                reason = f"{len(detections)}_detections"
                logger.info(f"Error case {label_path.name}: {len(detections)} detections (need exactly 11)")
                
                if not dry_run:
                    copy_to_error_folder(image_path, label_path, error_images_dir, error_labels_dir, reason, logger)
                
                stats["labels_copied_to_error"] += 1
                if reason not in stats["error_reasons"]:
                    stats["error_reasons"][reason] = 0
                stats["error_reasons"][reason] += 1
                processed_count += 1
                continue
            
            stats["labels_with_11_detections"] += 1
            
            # Extract container code from filename
            container_code = extract_container_code(image_name)
            
            if len(container_code) < 11:
                reason = "short_container_code"
                logger.info(f"Error case {image_name}: container code too short '{container_code}' (need 11 chars)")
                
                if not dry_run:
                    copy_to_error_folder(image_path, label_path, error_images_dir, error_labels_dir, reason, logger)
                
                stats["labels_copied_to_error"] += 1
                if reason not in stats["error_reasons"]:
                    stats["error_reasons"][reason] = 0
                stats["error_reasons"][reason] += 1
                processed_count += 1
                continue
            
            logger.info(f"Processing {image_name} with container code: {container_code}")
            
            # Remove overlapping detections (keep larger boxes)
            filtered_detections = remove_overlapping_detections(detections, overlap_threshold=0.5)
            
            # Log filtering info
            if len(filtered_detections) != len(detections):
                logger.info(f"  Removed {len(detections) - len(filtered_detections)} overlapping detections")
                logger.info(f"  Detections after overlap removal: {len(filtered_detections)}")
            
            # Find the best character sequence (handles tilted text and irrelevant detections)
            sequence_detections = find_character_sequence(filtered_detections, target_length=11)
            
            # Log sequence finding info
            if len(sequence_detections) != len(filtered_detections):
                logger.info(f"  Found best sequence: {len(sequence_detections)} from {len(filtered_detections)} detections")
            
            # Check if we have exactly 11 detections in the sequence
            if len(sequence_detections) != 11:
                reason = f"{len(sequence_detections)}_in_best_sequence"
                logger.info(f"Error case {image_name}: {len(sequence_detections)} detections in best sequence (need 11)")
                
                if not dry_run:
                    # Apply bounding box adjustment to the found sequence before saving to error folder
                    adjusted_sequence = adjust_overlapping_boxes(sequence_detections) if sequence_detections else []
                    copy_to_error_folder(image_path, label_path, error_images_dir, error_labels_dir, reason, logger, adjusted_sequence)
                
                stats["labels_copied_to_error"] += 1
                if reason not in stats["error_reasons"]:
                    stats["error_reasons"][reason] = 0
                stats["error_reasons"][reason] += 1
                processed_count += 1
                continue
            
            # Detections are already sorted by the sequence finding algorithm
            sorted_detections = sequence_detections
            
            # Adjust overlapping bounding boxes to prevent overlap
            adjusted_detections = adjust_overlapping_boxes(sorted_detections)
            
            # Map to OCR classes
            mapped_detections = map_detections_to_ocr(adjusted_detections, container_code)
            
            if len(mapped_detections) != 11:
                reason = "mapping_failed"
                logger.info(f"Error case {image_name}: mapping resulted in {len(mapped_detections)} detections (need 11)")
                
                if not dry_run:
                    # Use the adjusted detections (before OCR mapping) for the error folder
                    copy_to_error_folder(image_path, label_path, error_images_dir, error_labels_dir, reason, logger, adjusted_detections)
                
                stats["labels_copied_to_error"] += 1
                if reason not in stats["error_reasons"]:
                    stats["error_reasons"][reason] = 0
                stats["error_reasons"][reason] += 1
                processed_count += 1
                continue
            
            # Update character distribution
            for i, char in enumerate(container_code[:11]):
                if char not in stats["character_distribution"]:
                    stats["character_distribution"][char] = 0
                stats["character_distribution"][char] += 1
            
            if not dry_run:
                # Copy image to output
                output_image_path = output_images_dir / image_name
                shutil.copy2(image_path, output_image_path)
                
                # Write mapped label
                output_label_path = output_labels_dir / label_path.name
                write_yolo_label(output_label_path, mapped_detections)
                
                logger.debug(f"Copied {image_name} and created OCR label")
            
            stats["labels_mapped"] += 1
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {label_path.name}: {e}")
            stats["failed_labels"] += 1
            continue
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print processing statistics."""
    logger.info("\n=== PROCESSING STATISTICS ===")
    logger.info(f"Total label files: {stats['total_labels']}")
    logger.info(f"Labels processed: {stats['labels_processed']}")
    logger.info(f"Labels with 11 detections: {stats['labels_with_11_detections']}")
    logger.info(f"Labels successfully mapped: {stats['labels_mapped']}")
    logger.info(f"Labels copied to error folders: {stats['labels_copied_to_error']}")
    logger.info(f"Labels skipped: {stats['labels_skipped']}")
    logger.info(f"Failed labels: {stats['failed_labels']}")
    
    if stats['labels_processed'] > 0:
        success_rate = stats['labels_mapped'] / stats['labels_processed'] * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    if stats['labels_with_11_detections'] > 0:
        mapping_rate = stats['labels_mapped'] / stats['labels_with_11_detections'] * 100
        logger.info(f"Mapping success rate (11-detection labels): {mapping_rate:.1f}%")
    
    # Error reasons breakdown
    if stats['error_reasons']:
        logger.info("\nError reasons breakdown:")
        for reason, count in sorted(stats['error_reasons'].items()):
            logger.info(f"  {reason}: {count}")
    
    # Character distribution
    if stats['character_distribution']:
        logger.info("\nCharacter distribution in mapped labels:")
        sorted_chars = sorted(stats['character_distribution'].items())
        for char, count in sorted_chars:
            logger.info(f"  {char}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Map character detections to OCR labels based on filename",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=CHARACTERS_DIR / "images",
        help="Directory containing character images"
    )
    
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=CHARACTERS_DIR / "labels",
        help="Directory containing character labels"
    )
    
    parser.add_argument(
        "--output-images-dir",
        type=Path,
        default=OCR_OUTPUT_DIR / "images",
        help="Output directory for OCR images"
    )
    
    parser.add_argument(
        "--output-labels-dir",
        type=Path,
        default=OCR_OUTPUT_DIR / "labels",
        help="Output directory for OCR labels"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without copying files or creating labels"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Character to OCR Mapping Script ===")
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Labels directory: {args.labels_dir}")
    logger.info(f"Output images directory: {args.output_images_dir}")
    logger.info(f"Output labels directory: {args.output_labels_dir}")
    logger.info(f"Process limit: {args.limit if args.limit else 'None (process all)'}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Validate input paths
    if not args.images_dir.exists():
        logger.error(f"Images directory not found: {args.images_dir}")
        return 1
    
    if not args.labels_dir.exists():
        logger.error(f"Labels directory not found: {args.labels_dir}")
        return 1
    
    try:
        # Process images
        logger.info("\nStarting character to OCR mapping...")
        stats = process_images(
            args.images_dir, args.labels_dir,
            args.output_images_dir, args.output_labels_dir,
            args.limit, args.dry_run, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        if args.dry_run:
            logger.info("\n✅ Dry run completed successfully!")
            logger.info("Run without --dry-run to actually copy files and create labels")
        else:
            logger.info("\n✅ Character to OCR mapping completed successfully!")
            logger.info(f"OCR images saved to: {args.output_images_dir}")
            logger.info(f"OCR labels saved to: {args.output_labels_dir}")
            if stats['labels_copied_to_error'] > 0:
                error_base_dir = Path("data/detection/training_data/04_ocr_error")
                logger.info(f"Error images saved to: {error_base_dir / 'images'}")
                logger.info(f"Error labels saved to: {error_base_dir / 'labels'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())