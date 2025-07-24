#!/usr/bin/env python3
"""
Download Extra Container Images from CSV for Training Data

This script downloads images from URLs listed in extracontainers.csv and organizes them
based on whether the URL contains "cropped":
- URLs containing "cropped" → data/detection/training_data/10_raw/code_container_h
- All other URLs → data/detection/training_data/10_raw/code_container_tbd

Input: CSV file with columns "url" and "reference"

Usage:
    python3 data/detection/scripts/download_extracontainers.py
    
    # With options
    python3 data/detection/scripts/download_extracontainers.py --limit 100
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Default paths
DEFAULT_CSV_FILE = Path("data/detection/training_data/extracontainers.csv")
CROPPED_OUTPUT_DIR = Path("data/detection/training_data/10_raw/code_container_h")
TBD_OUTPUT_DIR = Path("data/detection/training_data/10_raw/code_container_tbd")
LOG_DIR = Path("data/detection/logs")

# Configuration
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
DELAY_BETWEEN_REQUESTS = 0.1  # seconds to be respectful

# Set up logging
def setup_logging(log_filename: str = "download_extracontainers.log"):
    """Configure logging to both file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / log_filename
    
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

def normalize_container_reference(reference: str) -> str:
    """
    Normalize container reference according to ISO 6346 standard.
    
    Args:
        reference: Raw reference string (may have quotes)
        
    Returns:
        Normalized reference with 0→O correction in first 3 characters
    """
    # Remove quotes if present
    normalized = reference.strip('\'"')
    
    # Convert to uppercase for consistency
    normalized = normalized.upper()
    
    # Replace 0 with O in first 3 characters (ISO 6346: first 4 chars are letters)
    if len(normalized) >= 3:
        prefix = normalized[:3]
        corrected_prefix = prefix.replace("0", "O")
        normalized = corrected_prefix + normalized[3:]
    
    return normalized

def download_image(url: str, output_path: Path, logger: logging.Logger, timeout: int = REQUEST_TIMEOUT) -> bool:
    """
    Download image from URL with retry logic.
    
    Args:
        url: Image URL to download
        output_path: Path where to save the image
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Create request with user agent to avoid blocking
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ContainerDataCollector/1.0)"
                }
            )
            
            with urlopen(request, timeout=timeout) as response:
                if response.status == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.read())
                    return True
                    
        except (URLError, HTTPError) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying...")
                time.sleep(1)  # Wait before retry
            else:
                logger.error(f"Failed to download {url} after {RETRY_ATTEMPTS} attempts: {e}")
                
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            break
    
    return False

def process_csv_file(csv_file: Path, limit: Optional[int], logger: logging.Logger, timeout: int = REQUEST_TIMEOUT):
    """
    Process CSV file and download container images.
    
    Args:
        csv_file: Path to CSV file with URLs and references
        limit: Maximum number of images to download (None for all)
        logger: Logger instance
    """
    # Ensure output directories exist
    CROPPED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TBD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        "total_rows": 0,
        "downloaded": 0,
        "downloaded_cropped": 0,
        "downloaded_tbd": 0,
        "skipped_existing": 0,
        "failed": 0,
        "invalid_reference": 0
    }
    
    # Track processed references to avoid duplicates
    processed_refs = set()
    
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            
            # Validate CSV headers
            if "url" not in csv_reader.fieldnames or "reference" not in csv_reader.fieldnames:
                logger.error(f"CSV file must have 'url' and 'reference' columns. Found: {csv_reader.fieldnames}")
                return
            
            logger.info(f"Starting download from {csv_file}")
            logger.info(f"Output directories:")
            logger.info(f"  - Cropped images: {CROPPED_OUTPUT_DIR}")
            logger.info(f"  - Other images: {TBD_OUTPUT_DIR}")
            if limit:
                logger.info(f"Download limit: {limit} images")
            
            for row_num, row in enumerate(csv_reader, 1):
                stats["total_rows"] += 1
                
                # Check limit
                if limit and stats["downloaded"] >= limit:
                    logger.info(f"Reached download limit of {limit} images")
                    break
                
                # Extract and validate data
                url = row.get("url", "").strip()
                raw_reference = row.get("reference", "").strip()
                
                if not url or not raw_reference:
                    logger.warning(f"Row {row_num}: Missing URL or reference, skipping")
                    stats["invalid_reference"] += 1
                    continue
                
                # Normalize reference
                try:
                    reference = normalize_container_reference(raw_reference)
                    
                    if len(reference) < 4:
                        logger.warning(f"Row {row_num}: Reference too short '{reference}', skipping")
                        stats["invalid_reference"] += 1
                        continue
                        
                except Exception as e:
                    logger.error(f"Row {row_num}: Error processing reference '{raw_reference}': {e}")
                    stats["invalid_reference"] += 1
                    continue
                
                # Check for duplicates
                if reference in processed_refs:
                    logger.debug(f"Row {row_num}: Duplicate reference '{reference}', skipping")
                    stats["skipped_existing"] += 1
                    continue
                
                # Determine output directory based on URL content
                is_cropped = "cropped" in url.lower()
                output_dir = CROPPED_OUTPUT_DIR if is_cropped else TBD_OUTPUT_DIR
                
                # Determine output filename
                output_filename = f"{reference}.jpg"
                output_path = output_dir / output_filename
                
                # Skip if file already exists
                if output_path.exists():
                    logger.debug(f"Row {row_num}: File already exists '{output_filename}', skipping")
                    stats["skipped_existing"] += 1
                    processed_refs.add(reference)
                    continue
                
                # Download image
                image_type = "cropped" if is_cropped else "tbd"
                logger.info(f"Row {row_num}: Downloading {reference} ({image_type}) from {url}")
                
                if download_image(url, output_path, logger, timeout):
                    stats["downloaded"] += 1
                    if is_cropped:
                        stats["downloaded_cropped"] += 1
                    else:
                        stats["downloaded_tbd"] += 1
                    processed_refs.add(reference)
                    logger.info(f"Row {row_num}: ✓ Saved {output_filename} to {output_dir.name}")
                else:
                    stats["failed"] += 1
                    logger.error(f"Row {row_num}: ✗ Failed to download {reference}")
                
                # Brief delay to be respectful to servers
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
                # Progress update every 50 downloads
                if stats["downloaded"] % 50 == 0:
                    logger.info(f"Progress: {stats['downloaded']} downloaded ({stats['downloaded_cropped']} cropped, {stats['downloaded_tbd']} tbd), {stats['failed']} failed")
    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        return
    
    # Final statistics
    logger.info("\n" + "="*50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*50)
    logger.info(f"Total rows processed: {stats['total_rows']}")
    logger.info(f"Successfully downloaded: {stats['downloaded']}")
    logger.info(f"  - Cropped images: {stats['downloaded_cropped']} → {CROPPED_OUTPUT_DIR.name}")
    logger.info(f"  - TBD images: {stats['downloaded_tbd']} → {TBD_OUTPUT_DIR.name}")
    logger.info(f"Skipped (already exists): {stats['skipped_existing']}")
    logger.info(f"Failed downloads: {stats['failed']}")
    logger.info(f"Invalid references: {stats['invalid_reference']}")
    
    if stats["downloaded"] > 0:
        logger.info(f"\n✓ Successfully downloaded {stats['downloaded']} container images!")
    
    if stats["failed"] > 0:
        logger.warning(f"\n⚠ {stats['failed']} downloads failed - check log for details")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download extra container images from CSV file and organize by type",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=DEFAULT_CSV_FILE,
        help="Path to CSV file with container URLs and references"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of images to download (for testing)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help="Request timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Validate input file
    if not args.csv_file.exists():
        logger.error(f"CSV file not found: {args.csv_file}")
        logger.info(f"Expected format: CSV with 'url' and 'reference' columns")
        return 1
    
    # Start processing
    logger.info("Extra Container Image Downloader Starting...")
    start_time = time.time()
    
    try:
        process_csv_file(args.csv_file, args.limit, logger, args.timeout)
    
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\nTotal time: {elapsed:.1f} seconds")
    
    return 0

if __name__ == "__main__":
    exit(main())