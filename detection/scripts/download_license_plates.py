#!/usr/bin/env python3
"""
Download License Plate Images from CSV for Training Data

This script downloads license plate images from URLs listed in a CSV file and saves them
with filename-safe normalized names in the training data structure.

Input: code_license_plate.csv with columns "url" and "reference"

The input file is generated using this SQL query:
SELECT top 2000
	('https://media.cargosnap.net/thumbnails/sm/tenant' + CONVERT(varchar(4), s.tenant_id) + '/' + s.image_path) as url,
	f.value as reference
FROM	
	cargosnapdb.dbo.snaps s
	LEFT JOIN files_tags f on f.snap_id = s.id 
WHERE 
	s.id > 120000000
	and s.workflow_step_id in (
        SELECT
            wfs.id
        FROM
            cargosnapdb.dbo.workflows_steps wfs
        JOIN
            cargosnapdb.dbo.tenants t on t.id = wfs.tenant_id
        WHERE
            wfs.deleted_at IS NULL
            AND wfs.[type] = 'licensePlateScanner'
            AND t.id NOT IN (1,20)
            AND t.status IN ('standard', 'enterprise')
    )
	and s.deleted_at is null
	and f.tenant_id NOT IN (1,20)



Usage Examples:
    # Download license plates (default)
    python3 detection/scripts/download_license_plates.py
    
    # Download with custom settings
    python3 detection/scripts/download_license_plates.py \\
        --csv-file path/to/custom.csv \\
        --output-dir detection/training_data/00_raw/code_license_plate \\
        --limit 100

    # Download with specific timeout
    python3 detection/scripts/download_license_plates.py \\
        --timeout 60

The script:
1. Reads CSV file with license plate image URLs and references
2. Processes references by removing quotes and making filenames safe
3. Downloads images and saves them as {safe_reference}.jpg
4. Skips duplicates and reports download status
5. Creates progress tracking and error logs

Note: Special characters in references are replaced with underscores for safe filenames
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
DEFAULT_CSV_FILE = "detection/training_data/00_raw/code_license_plate.csv"
DEFAULT_OUTPUT_DIR = Path("detection/training_data/00_raw/code_license_plate")
LOG_DIR = Path("detection/logs")

# Configuration
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
DELAY_BETWEEN_REQUESTS = 0.1  # seconds to be respectful

# Set up logging
def setup_logging(log_filename: str = "download_images.log"):
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

def make_filename_safe(reference: str) -> str:
    """
    Make a reference filename-safe by replacing special characters with underscores.
    
    Args:
        reference: Raw reference string (may have quotes or special chars)
        
    Returns:
        Filename-safe reference with special characters replaced by underscores
    """
    import re
    
    # Remove quotes if present
    cleaned = reference.strip('\'"')
    
    # Convert to uppercase for consistency
    cleaned = cleaned.upper()
    
    # Replace any character that's not alphanumeric or hyphen with underscore
    # Keep only: A-Z, 0-9, hyphen, underscore
    safe_filename = re.sub(r'[^A-Z0-9\-_]', '_', cleaned)
    
    # Remove multiple consecutive underscores
    safe_filename = re.sub(r'_+', '_', safe_filename)
    
    # Remove leading/trailing underscores
    safe_filename = safe_filename.strip('_')
    
    # Ensure we have something (fallback to "unknown" if empty)
    if not safe_filename:
        safe_filename = "unknown"
    
    return safe_filename

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
                    "User-Agent": "Mozilla/5.0 (compatible; LicensePlateDataCollector/1.0)"
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

def process_csv_file(csv_file: Path, output_dir: Path, limit: Optional[int], logger: logging.Logger, timeout: int = REQUEST_TIMEOUT):
    """
    Process CSV file and download license plate images.
    
    Args:
        csv_file: Path to CSV file with URLs and references
        output_dir: Directory to save downloaded images
        limit: Maximum number of images to download (None for all)
        logger: Logger instance
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        "total_rows": 0,
        "downloaded": 0,
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
            logger.info(f"Output directory: {output_dir}")
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
                
                # Make reference filename-safe
                try:
                    reference = make_filename_safe(raw_reference)
                    
                    if len(reference) < 1:
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
                logger.info(f"Row {row_num}: Downloading {reference} from {url}")
                
                if download_image(url, output_path, logger, timeout):
                    stats["downloaded"] += 1
                    processed_refs.add(reference)
                    logger.info(f"Row {row_num}: ✓ Saved {output_filename}")
                else:
                    stats["failed"] += 1
                    logger.error(f"Row {row_num}: ✗ Failed to download {reference}")
                
                # Brief delay to be respectful to servers
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
                # Progress update every 50 downloads
                if stats["downloaded"] % 50 == 0:
                    logger.info(f"Progress: {stats['downloaded']} downloaded, {stats['failed']} failed")
    
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
    logger.info(f"Skipped (already exists): {stats['skipped_existing']}")
    logger.info(f"Failed downloads: {stats['failed']}")
    logger.info(f"Invalid references: {stats['invalid_reference']}")
    logger.info(f"Output directory: {output_dir}")
    
    if stats["downloaded"] > 0:
        logger.info(f"\n✓ Successfully downloaded {stats['downloaded']} license plate images!")
    
    if stats["failed"] > 0:
        logger.warning(f"\n⚠ {stats['failed']} downloads failed - check log for details")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download license plate images from CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=DEFAULT_CSV_FILE,
        help="Path to CSV file with license plate URLs and references"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save downloaded images"
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
    
    # Set up logging with dynamic filename based on output directory
    output_name = args.output_dir.name
    log_filename = f"download_{output_name}.log"
    logger = setup_logging(log_filename)
    
    # Validate input file
    if not args.csv_file.exists():
        logger.error(f"CSV file not found: {args.csv_file}")
        logger.info(f"Expected format: CSV with 'url' and 'reference' columns")
        return 1
    
    # Start processing
    logger.info("License Plate Image Downloader Starting...")
    start_time = time.time()
    
    try:
        process_csv_file(args.csv_file, args.output_dir, args.limit, logger, args.timeout)
    
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