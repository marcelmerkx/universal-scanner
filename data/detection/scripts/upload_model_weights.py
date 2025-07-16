#!/usr/bin/env python3
"""
Upload model weights to Roboflow project.

This script uploads a trained PyTorch YOLO model to a Roboflow project
for deployment and inference.

Usage:
    python3 data/detection/scripts/upload_model_weights.py
    python3 data/detection/scripts/upload_model_weights.py --model-name "detection-v1"
    python3 data/detection/scripts/upload_model_weights.py --model-path path/to/model.pt
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed. Run: pip install roboflow")
    exit(1)

# Configuration
DEFAULT_MODEL_PATH = Path("data/detection/models/detection_320_grayscale_tilted-09-07-2025.pt")
LOG_DIR = Path("data/detection/logs")

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / '.env')


def setup_logging():
    """Configure logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "upload_model_weights.log"
    
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


def upload_model_weights(api_key: str, workspace: str, project_id: str,
                        model_path: Path, model_name: Optional[str], 
                        version: Optional[int], logger: logging.Logger) -> Dict:
    """
    Upload model weights to Roboflow project.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Roboflow project ID
        model_path: Path to the model weights file (.pt)
        model_name: Optional custom name for the deployed model
        version: Optional specific dataset version to deploy to
        logger: Logger instance
        
    Returns:
        Upload statistics and result info
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Validate model file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.suffix == '.pt':
        raise ValueError(f"Model file must be a .pt file, got '{model_path.suffix}'")
    
    # Extract model name if not provided
    if not model_name:
        model_name = model_path.stem
    
    # Ensure model name has at least one letter (Roboflow requirement)
    if not any(c.isalpha() for c in model_name):
        model_name = f"model_{model_name}"
    
    logger.info(f"Model file: {model_path}")
    logger.info(f"Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    logger.info(f"Model name: {model_name}")
    
    stats = {
        "model_path": str(model_path),
        "model_name": model_name,
        "model_size_mb": model_path.stat().st_size / (1024*1024),
        "upload_successful": False,
        "upload_time": 0.0,
        "deployment_info": None,
        "error": None
    }
    
    start_time = time.time()
    
    try:
        logger.info("Starting model deployment to Roboflow...")
        
        # Get the project
        project = rf.workspace(workspace).project(project_id)
        
        if version is not None:
            # Deploy to specific version
            logger.info(f"Deploying to specific version: {version}")
            dataset_version = project.version(version)
        else:
            # Get the latest version
            versions = project.versions()
            if not versions:
                raise Exception("No dataset versions found in project. Please create a dataset version first.")
            
            # Get the most recent version number
            latest_version_num = max(v.version for v in versions)
            logger.info(f"Using latest dataset version: {latest_version_num}")
            dataset_version = project.version(latest_version_num)
        
        # Deploy the model following the documentation
        logger.info(f"Deploying model to version {dataset_version.version}...")
        
        deployment_result = dataset_version.deploy(
            model_type="yolov8",
            model_path=str(model_path.parent),  # Directory path
            filename=model_path.name  # Just the filename
        )
        
        stats["upload_successful"] = True
        stats["deployment_info"] = deployment_result
        logger.info(f"✅ Successfully deployed model '{model_name}' to project!")
        
        # Log deployment details if available
        if deployment_result:
            logger.info(f"Deployment result: {deployment_result}")
        
    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"Model deployment failed: {e}")
        raise
    
    finally:
        stats["upload_time"] = time.time() - start_time
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    """Print upload statistics."""
    logger.info("\n=== MODEL UPLOAD STATISTICS ===")
    logger.info(f"Model file: {stats['model_path']}")
    logger.info(f"Model name: {stats['model_name']}")
    logger.info(f"Model size: {stats['model_size_mb']:.1f} MB")
    logger.info(f"Upload time: {stats['upload_time']:.1f}s")
    logger.info(f"Upload successful: {stats['upload_successful']}")
    
    if stats['upload_successful'] and stats['deployment_info']:
        logger.info(f"Deployment info: {stats['deployment_info']}")
    
    if stats['error']:
        logger.error(f"Error: {stats['error']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload model weights to Roboflow project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--api-key",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
        default=os.environ.get("ROBOFLOW_API_KEY")
    )
    
    parser.add_argument(
        "--workspace",
        default="cargosnap",
        help="Roboflow workspace name"
    )
    
    parser.add_argument(
        "--project",
        default="unified-detection-0zmvz",
        help="Roboflow project ID"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model weights file (.pt)"
    )
    
    parser.add_argument(
        "--model-name",
        help="Custom name for the deployed model (default: uses filename)"
    )
    
    parser.add_argument(
        "--version",
        type=int,
        help="Specific dataset version to deploy to (default: latest version)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files but don't actually upload to Roboflow"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("=== Upload Model Weights to Roboflow ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Project: {args.project}")
    logger.info(f"Model name: {args.model_name if args.model_name else 'auto-generated from filename'}")
    logger.info(f"Version: {args.version if args.version else 'latest'}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Check if API key is provided
    if not args.api_key and not args.dry_run:
        logger.error("Error: Roboflow API key is required.")
        logger.error("Provide it via --api-key or set ROBOFLOW_API_KEY environment variable.")
        return 1
    
    # Validate model file
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    if not args.model_path.suffix == '.pt':
        logger.error(f"Model file must be a .pt file, got '{args.model_path.suffix}'")
        return 1
    
    try:
        if args.dry_run:
            logger.info(f"\n✅ Dry run completed!")
            logger.info(f"Model file validated: {args.model_path}")
            logger.info(f"Model size: {args.model_path.stat().st_size / (1024*1024):.1f} MB")
            
            model_name = args.model_name if args.model_name else args.model_path.stem
            if not any(c.isalpha() for c in model_name):
                model_name = f"model_{model_name}"
            
            logger.info(f"Would deploy with name: {model_name}")
            logger.info("Use --api-key and remove --dry-run to perform actual upload")
            return 0
        
        # Upload model
        logger.info("\nStarting model upload to Roboflow...")
        
        stats = upload_model_weights(
            args.api_key, args.workspace, args.project,
            args.model_path, args.model_name, args.version, logger
        )
        
        # Print statistics
        print_statistics(stats, logger)
        
        if stats["upload_successful"]:
            logger.info(f"\n✅ Model upload completed successfully!")
            logger.info(f"Model '{stats['model_name']}' is now available in your Roboflow project")
            logger.info(f"Check your project: https://app.roboflow.com/{args.workspace}/{args.project}")
        else:
            logger.error(f"\n❌ Model upload failed!")
            if stats["error"]:
                logger.error(f"Error details: {stats['error']}")
        
        return 0 if stats["upload_successful"] else 1
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())