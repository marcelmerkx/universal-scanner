#!/usr/bin/env python3
"""Debug script to check label matching"""

from pathlib import Path

# Paths
labels_dir = Path("/Users/marcelmerkx/Development/universal-scanner/data/OCR/labels")
cutouts_dir = Path("/Users/marcelmerkx/Development/universal-scanner/data/OCR/cutouts")

# Get all label files
label_files = list(labels_dir.glob("*.txt"))
print(f"Found {len(label_files)} label files")

# Check for matching images
matched = 0
unmatched = []

for label_file in label_files:
    if label_file.name == "labeling_stats.json":
        continue
        
    base_name = label_file.stem
    
    # Look for corresponding image
    found = False
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_path = cutouts_dir / f"{base_name}{ext}"
        if image_path.exists():
            print(f"✓ {label_file.name} -> {image_path.name}")
            matched += 1
            found = True
            break
    
    if not found:
        print(f"✗ {label_file.name} -> No matching image found")
        unmatched.append(base_name)

print(f"\nSummary:")
print(f"Matched: {matched}")
print(f"Unmatched: {len(unmatched)}")

if unmatched:
    print(f"\nUnmatched labels: {unmatched}")