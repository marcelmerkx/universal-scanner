#!/usr/bin/env python3
"""
Upload images to Roboflow project.

This script uploads cutout images from the local dataset to a Roboflow project.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


def upload_images_to_roboflow(api_key, workspace, project_id, images_path):
    """
    Upload images from a directory to a Roboflow project.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Roboflow project ID
        images_path: Path to directory containing images
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get workspace and project
    workspace_obj = rf.workspace(workspace)
    project = workspace_obj.project(project_id)
    
    # Get all image files from the directory
    images_dir = Path(images_path)
    if not images_dir.exists():
        print(f"Error: Images directory '{images_path}' does not exist.")
        sys.exit(1)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in '{images_path}'")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to upload")
    
    # Upload images
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"Uploading {i}/{len(image_files)}: {image_file.name}")
            project.upload(str(image_file))
        except Exception as e:
            print(f"Error uploading {image_file.name}: {e}")
            continue
    
    print("Upload completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Upload images to Roboflow project"
    )
    parser.add_argument(
        "--api-key",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
        default=os.environ.get("ROBOFLOW_API_KEY")
    )
    parser.add_argument(
        "--workspace",
        default="cargosnap",
        help="Roboflow workspace name (default: cargosnap)"
    )
    parser.add_argument(
        "--project-id",
        default="horizontal-ocr",
        help="Roboflow project ID (default: horizontal-ocr)"
    )
    parser.add_argument(
        "--images-path",
        default="OCR/cutouts",
        help="Path to images directory (default: OCR/morecutouts)"
    )
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: Roboflow API key is required.")
        print("Provide it via --api-key or set ROBOFLOW_API_KEY environment variable.")
        sys.exit(1)
    
    # Make path absolute if it's relative
    if not os.path.isabs(args.images_path):
        args.images_path = os.path.abspath(args.images_path)
    
    # Upload images
    upload_images_to_roboflow(
        args.api_key,
        args.workspace,
        args.project_id,
        args.images_path
    )


if __name__ == "__main__":
    main()