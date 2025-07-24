#!/usr/bin/env python3
"""
Remove duplicate images from container_h directory where files with the same container code
already exist in either 02_characters/images or OCR/horizontal/images directories.
"""

import os
import sys
from pathlib import Path
import argparse


def extract_container_code(filename):
    """Extract the base container code from a filename, ignoring sequence numbers."""
    base = Path(filename).stem
    # Remove sequence numbers like -1, -2, etc.
    if '-' in base:
        base = base.split('-')[0]
    return base


def find_duplicates(source_dir, compare_dirs, dry_run=True):
    """Find and optionally remove duplicate images based on container codes."""
    
    # Convert to Path objects
    source_path = Path(source_dir)
    compare_paths = [Path(d) for d in compare_dirs]
    
    # Verify directories exist
    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist")
        return
    
    for path in compare_paths:
        if not path.exists():
            print(f"Warning: Compare directory {path} does not exist")
    
    # Collect all container codes from compare directories
    existing_codes = set()
    for compare_path in compare_paths:
        if compare_path.exists():
            for file in compare_path.glob("*.jpg"):
                code = extract_container_code(file.name)
                existing_codes.add(code)
    
    print(f"Found {len(existing_codes)} unique container codes in compare directories")
    
    # Check source directory for duplicates
    duplicates = []
    source_files = list(source_path.glob("*.jpg"))
    
    for file in source_files:
        code = extract_container_code(file.name)
        if code in existing_codes:
            duplicates.append(file)
    
    print(f"\nFound {len(duplicates)} duplicate files in {source_path}")
    
    if duplicates:
        print("\nDuplicate files:")
        for dup in sorted(duplicates):
            print(f"  {dup.name}")
        
        if dry_run:
            print("\n[DRY RUN] No files removed. Use --execute to actually remove files.")
        else:
            print("\nRemoving duplicate files...")
            removed_count = 0
            for dup in duplicates:
                try:
                    dup.unlink()
                    print(f"  Removed: {dup.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"  Error removing {dup.name}: {e}")
            
            print(f"\nRemoved {removed_count} duplicate files")
    else:
        print("No duplicates found!")
    
    print(f"\nRemaining files in source directory: {len(source_files) - len(duplicates)}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate container images based on container codes"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove files (default is dry run)"
    )
    parser.add_argument(
        "--source",
        default="data/detection/training_data/20_raw/container_h",
        help="Source directory to check for duplicates"
    )
    
    args = parser.parse_args()
    
    # Define directories relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # Go up three levels from scripts/ to project root
    
    source_dir = project_root / args.source
    compare_dirs = [
        project_root / "data/detection/training_data/02_characters/images",
        project_root / "data/OCR/horizontal/images"
    ]
    
    print(f"Source directory: {source_dir}")
    print(f"Compare directories:")
    for d in compare_dirs:
        print(f"  - {d}")
    print()
    
    find_duplicates(source_dir, compare_dirs, dry_run=not args.execute)


if __name__ == "__main__":
    main()