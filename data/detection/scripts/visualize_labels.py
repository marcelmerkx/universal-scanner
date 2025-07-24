#!/usr/bin/env python3
"""
Visualize labeled images with Back/Next navigation.

This script:
1. Loads images and labels from a folder structure
2. Displays images with bounding boxes overlaid
3. Allows navigation through the dataset with Back/Next
4. Shows class names and confidence information

Color Coding:
- Each character class (0-9, A-Z) has a consistent color
- Same characters will always have the same color across all images
- This makes it easy to spot labeling errors or patterns

Usage:
    python3 data/detection/scripts/visualize_labels.py
    python3 data/detection/scripts/visualize_labels.py --root data/detection/training_data/03_ocr_generated
    
Navigation:
    - Click Next/Previous buttons or use arrow keys
    - Press 'q' or ESC to quit
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches

# OCR class mapping from best-OCR-18-06-25-data.yaml
OCR_CLASS_NAMES = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z'
}

# Character detection class (fallback for single class models)
CHARACTER_CLASS_NAMES = {
    0: 'character'
}

# Color palette for bounding boxes
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 255, 0),  # Lime
    (255, 0, 128),  # Pink
    (128, 0, 255),  # Purple
]


class LabelVisualizer:
    def __init__(self, root_folder: Path):
        self.root_folder = root_folder
        self.current_index = 0
        self.image_files = []
        self.fig = None
        self.ax = None
        
        # Find images and labels
        self.images_dir, self.labels_dir = self._find_dirs()
        self.image_files = self._find_image_files()
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"Found {len(self.image_files)} images")
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")
    
    def _find_dirs(self) -> Tuple[Path, Path]:
        """Find images and labels directories."""
        # Check if root folder directly contains images
        if any(f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] 
               for f in self.root_folder.iterdir() if f.is_file()):
            # Root folder contains images directly
            images_dir = self.root_folder
            # Look for labels in parent or sibling directory
            if (self.root_folder.parent / "labels").exists():
                labels_dir = self.root_folder.parent / "labels"
            else:
                labels_dir = self.root_folder  # Assume labels are in same folder
        else:
            # Look for standard structure
            images_dir = self.root_folder / "images"
            labels_dir = self.root_folder / "labels"
            
            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
        
        return images_dir, labels_dir
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = sorted([f for f in self.images_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions])
        return files
    
    def _parse_yolo_label(self, label_path: Path) -> List[Dict]:
        """Parse YOLO format label file."""
        detections = []
        
        if not label_path.exists():
            return detections
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid label format in {label_path.name} line {line_num}: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        detections.append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': width,
                            'height': height
                        })
                    except ValueError as e:
                        print(f"Warning: Could not parse line {line_num} in {label_path.name}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
        
        return detections
    
    def _yolo_to_pixel(self, detection: Dict, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert YOLO format to pixel coordinates."""
        center_x = detection['center_x'] * img_width
        center_y = detection['center_y'] * img_height
        width = detection['width'] * img_width
        height = detection['height'] * img_height
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        return x1, y1, x2, y2
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name for given class ID."""
        # Try OCR classes first
        if class_id in OCR_CLASS_NAMES:
            return OCR_CLASS_NAMES[class_id]
        
        # Try character classes
        if class_id in CHARACTER_CLASS_NAMES:
            return CHARACTER_CLASS_NAMES[class_id]
        
        # Default to class ID
        return f"class_{class_id}"
    
    def _display_image(self):
        """Display current image with labels."""
        if not self.image_files:
            return
        
        # Clear current plot
        if self.ax:
            self.ax.clear()
        
        # Load current image
        image_path = self.image_files[self.current_index]
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Load corresponding label
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        detections = self._parse_yolo_label(label_path)
        
        # Display image
        self.ax.imshow(image_rgb)
        self.ax.set_title(f"Image {self.current_index + 1}/{len(self.image_files)}: {image_path.name}\n"
                         f"Detections: {len(detections)}", fontsize=12, pad=20)
        
        # Draw bounding boxes
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = self._yolo_to_pixel(detection, img_width, img_height)
            
            # Get color based on class_id so same characters have same color
            color = np.array(COLORS[detection['class_id'] % len(COLORS)]) / 255.0  # Normalize to [0,1]
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            self.ax.add_patch(rect)
            
            # Add label
            class_name = self._get_class_name(detection['class_id'])
            label_text = f"{class_name}"
            
            # Label background
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
            self.ax.text(x1, y1 - 5, label_text, fontsize=10, color='white',
                        bbox=bbox_props, verticalalignment='bottom')
        
        # Show detection info in bottom
        if detections:
            # Sort detections by x-coordinate for display
            sorted_detections = sorted(detections, key=lambda d: d['center_x'])
            
            # Create string of detected characters/classes
            detected_chars = []
            for det in sorted_detections:
                char = self._get_class_name(det['class_id'])
                detected_chars.append(char)
            
            detected_string = ''.join(detected_chars)
            self.ax.text(0.02, 0.02, f"Detected sequence: {detected_string}", 
                        transform=self.ax.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        self.ax.axis('off')
        
        # Update title
        container_code = image_path.stem
        if '-' in container_code:
            container_code = container_code.split('-')[0]  # Remove suffix
        
        main_title = f"Label Visualizer - {self.root_folder.name}"
        if len(detections) == 11:
            main_title += f" | Expected: {container_code[:11]}"
        
        self.fig.suptitle(main_title, fontsize=14, y=0.98)
        
        # Refresh display
        self.fig.canvas.draw()
    
    def _next_image(self, event):
        """Go to next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._display_image()
        else:
            print("Already at last image")
    
    def _prev_image(self, event):
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_image()
        else:
            print("Already at first image")
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'right' or event.key == 'n':
            self._next_image(event)
        elif event.key == 'left' or event.key == 'p':
            self._prev_image(event)
        elif event.key == 'q' or event.key == 'escape':
            plt.close('all')
            sys.exit(0)
    
    def show(self):
        """Start the visualization."""
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.05])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        
        btn_prev.on_clicked(self._prev_image)
        btn_next.on_clicked(self._next_image)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Display first image
        self._display_image()
        
        # Add instructions
        instructions = ("Navigation: ←/→ arrows or 'p'/'n' keys | 'q' or ESC to quit")
        self.fig.text(0.5, 0.01, instructions, ha='center', fontsize=10, style='italic')
        
        # Show plot
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize labeled images with navigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/detection/training_data/03_ocr_generated"),
        help="Root folder containing images and labels"
    )
    
    args = parser.parse_args()
    
    # Validate root folder
    if not args.root.exists():
        print(f"Error: Root folder not found: {args.root}")
        return 1
    
    try:
        # Create and run visualizer
        visualizer = LabelVisualizer(args.root)
        visualizer.show()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())