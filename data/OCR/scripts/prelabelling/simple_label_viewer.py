#!/usr/bin/env python3
"""
Simple YOLO Label Viewer
A straightforward script to view YOLO labels overlaid on images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
try:
    from config_template import INPUT_DIR, OUTPUT_DIR
except ImportError:
    INPUT_DIR = "/Users/marcelmerkx/Development/universal-scanner/data/OCR/cutouts"
    OUTPUT_DIR = "/Users/marcelmerkx/Development/universal-scanner/data/OCR/labels"

def read_yolo_labels(label_path):
    """Read YOLO format labels from file"""
    boxes = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
    
    return boxes

def yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO format to xyxy format"""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return x1, y1, x2, y2

def get_image_label_pairs():
    """Get all image files that have corresponding label files"""
    images_dir = Path(INPUT_DIR)
    labels_dir = Path(OUTPUT_DIR)
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    pairs = []
    
    # Get all label files
    label_files = list(labels_dir.glob("*.txt"))
    
    for label_file in sorted(label_files):
        # Look for corresponding image
        base_name = label_file.stem
        image_file = None
        
        for ext in extensions:
            potential_image = images_dir / f"{base_name}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break
            # Try uppercase extension
            potential_image = images_dir / f"{base_name}{ext.upper()}"
            if potential_image.exists():
                image_file = potential_image
                break
        
        if image_file:
            pairs.append((image_file, label_file))
    
    return pairs

def show_image_with_labels(image_path, label_path, ax=None):
    """Display an image with its YOLO labels"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Read labels
    boxes = read_yolo_labels(label_path)
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.imshow(image_rgb)
    
    # Draw boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(boxes))))
    
    for i, (box, color) in enumerate(zip(boxes, colors)):
        x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add box number
        ax.text(
            x1, y1 - 5, f'Box {i+1}',
            color=color, fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    ax.set_title(f"{image_path.name} | {len(boxes)} boxes")
    ax.axis('off')

def main():
    """Main function to display all labeled images"""
    pairs = get_image_label_pairs()
    print(f"Found {len(pairs)} labeled images")
    
    if not pairs:
        print("No labeled images found!")
        return
    
    # Create a simple interactive viewer
    current_index = 0
    
    def show_current():
        plt.clf()
        image_path, label_path = pairs[current_index]
        show_image_with_labels(image_path, label_path)
        plt.suptitle(f"Image {current_index + 1} of {len(pairs)} | Press 'n' for next, 'p' for previous, 'q' to quit")
        plt.draw()
    
    def on_key(event):
        nonlocal current_index
        
        if event.key == 'n' and current_index < len(pairs) - 1:
            current_index += 1
            show_current()
        elif event.key == 'p' and current_index > 0:
            current_index -= 1
            show_current()
        elif event.key == 'q':
            plt.close()
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show first image
    show_current()
    plt.show()
    
    print("\nAlternatively, showing all images in a grid...")
    
    # Show all in a grid
    n_images = len(pairs)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i, (image_path, label_path) in enumerate(pairs):
        if i < len(axes):
            show_image_with_labels(image_path, label_path, axes[i])
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()