#!/usr/bin/env python3
"""
Upload YOLOv8 model weights to Roboflow.

This script uploads a trained YOLOv8 model to a Roboflow project for deployment.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')


def upload_yolov8_model(api_key, workspace, project_id, model_path, model_name=None):
    """
    Upload YOLOv8 model weights to a Roboflow project.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Roboflow project ID
        model_path: Path to the model weights file (.pt)
        model_name: Optional custom name for the deployed model
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get workspace
    workspace_obj = rf.workspace(workspace)
    
    # Check if model file exists
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
    
    if not model_file.suffix == '.pt':
        print(f"Error: Model file must be a .pt file, got '{model_file.suffix}'")
        sys.exit(1)
    
    print(f"Uploading YOLOv8 model: {model_file}")
    print(f"Workspace: {workspace}")
    print(f"Project: {project_id}")
    
    # Extract model name if not provided
    if not model_name:
        model_name = model_file.stem
    
    try:
        # Deploy model to workspace level (versionless deployment)
        print(f"Deploying model '{model_name}' to workspace...")
        workspace_obj.deploy_model(
            model_type="yolov8",
            model_path=str(model_file.parent),  # Directory containing the model
            filename=model_file.name,  # Model filename
            project_ids=[project_id],  # Projects to deploy to
            model_name=model_name
        )
        print(f"Successfully deployed model '{model_name}' to project '{project_id}'!")
        
    except Exception as e:
        print(f"Error uploading model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload YOLOv8 model weights to Roboflow"
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
        "--model-path",
        default="ContainerCameraApp/assets/models/best-OCR-Colab-22-06-25.pt",
        help="Path to YOLOv8 model weights file (.pt)"
    )
    parser.add_argument(
        "--model-name",
        help="Custom name for the deployed model (default: uses filename)"
    )
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: Roboflow API key is required.")
        print("Provide it via --api-key or set ROBOFLOW_API_KEY environment variable.")
        sys.exit(1)
    
    # Make path absolute if it's relative
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.abspath(args.model_path)
    
    # Upload model
    upload_yolov8_model(
        args.api_key,
        args.workspace,
        args.project_id,
        args.model_path,
        args.model_name
    )


if __name__ == "__main__":
    main()