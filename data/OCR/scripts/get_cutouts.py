#!/usr/bin/env python3
"""
Image Downloader Script

This script reads a CSV file containing image URLs and their reference names,
then downloads these images to a specified cutouts directory. It's designed to handle
batch downloading of container images for further processing.

The CSV file should have at least two columns:
- url: The URL of the image to download
- reference: A unique identifier/name for the image

Usage:
    python3 OCR/scripts/get_cutouts.py

    (if not activated, run "source .venv/bin/activate && python3 scripts/get_cutouts.py")

Example:
    # Run from the project root directory
    python3 scripts/get_cutouts.py
    
    # This will read from: OCR/2000cutouts.csv
    # And download images to: OCR/cutouts/

Requirements:
    - pandas
    - requests
    - pathlib
"""

import os
import sys
from pathlib import Path
from typing import NoReturn
import logging
import pandas as pd
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - Edit these paths as needed
CSV_PATH = "OCR/morecutouts.csv"
OUTPUT_DIR = "OCR/morecutouts"

def setup_directories(output_dir: str) -> Path:
    """
    Create the output directory if it doesn't exist.
    
    Args:
        output_dir: The path where downloaded images will be stored
        
    Returns:
        Path: A Path object pointing to the created directory
        
    Raises:
        PermissionError: If the script lacks permission to create the directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def download_image(url: str, filepath: Path) -> bool:
    """
    Download an image from a URL and save it to the specified path.
    
    Args:
        url: The URL of the image to download
        filepath: The path where the image should be saved
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
        
    except RequestException as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False

def main() -> NoReturn:
    """
    Main function to orchestrate the image downloading process.
    Reads the CSV file and downloads all specified images.
    """
    try:
        # Setup output directory
        output_path = setup_directories(OUTPUT_DIR)
        
        # Read CSV file
        try:
            df = pd.read_csv(CSV_PATH)
            required_columns = {"url", "reference"}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"CSV is missing required columns: {missing}")
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error(f"Failed to read CSV file: {str(e)}")
            sys.exit(1)
            
        # Download images
        total_images = len(df)
        successful_downloads = 0
        
        for index, row in df.iterrows():
            image_path = output_path / f"{row['reference']}.jpg"
            logger.info(f"Downloading image {index + 1}/{total_images}: {row['reference']}")
            
            if download_image(row["url"], image_path):
                successful_downloads += 1
                
        # Report results
        logger.info(f"Download complete. Successfully downloaded {successful_downloads}/{total_images} images.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 