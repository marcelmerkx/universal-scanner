import os
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def train_model():
    """
    Train a YOLOv8 model on the container digit dataset.
    Uses early stopping and learning rate scheduling for efficiency.
    """
    try:
        # Initialize model
        logger.info("Initializing YOLOv8 model...")
        model = YOLO("yolov8n.pt")  # Load pretrained model

        # Training configuration
        epochs = 40  # Reduced from 100 to 40
        batch_size = 16
        img_size = 640
        device = "cpu"  # or "cuda" if GPU available

        logger.info("Starting training with configuration:")
        logger.info(f"- Epochs: {epochs}")
        logger.info(f"- Batch size: {batch_size}")
        logger.info(f"- Image size: {img_size}")
        logger.info(f"- Device: {device}")

        # Train the model with improved configuration
        results = model.train(
            data="data/dataset/dataset.yaml",
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            patience=15,  # Early stopping patience
            lr0=0.01,     # Initial learning rate
            lrf=0.01,     # Final learning rate
            warmup_epochs=3.0,  # Learning rate warmup
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,      # Box loss gain
            cls=0.5,      # Class loss gain
            dfl=1.5,      # DFL loss gain
            close_mosaic=10,  # Disable mosaic augmentation for last 10 epochs
            hsv_h=0.015,  # HSV augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,  # Rotation augmentation
            translate=0.1,  # Translation augmentation
            scale=0.5,    # Scale augmentation
            shear=0.0,    # Shear augmentation
            perspective=0.0,  # Perspective augmentation
            flipud=0.0,   # Flip up-down augmentation
            fliplr=0.5,   # Flip left-right augmentation
            mosaic=1.0,   # Mosaic augmentation
            mixup=0.0,    # Mixup augmentation
            copy_paste=0.0,  # Copy-paste augmentation
            verbose=True,  # Print verbose output
            seed=42,      # Random seed for reproducibility
            deterministic=True,  # Deterministic training
            single_cls=True,  # Single class detection
            rect=False,   # Rectangular training
            cos_lr=False,  # Cosine learning rate scheduler
            label_smoothing=0.0,  # Label smoothing
            nbs=64,       # Nominal batch size
            overlap_mask=True,  # Overlap masks during training
            mask_ratio=4,  # Mask downsample ratio
            dropout=0.0,  # Dropout regularization
            val=True,     # Validate during training
            plots=True,   # Generate plots
            save=True,    # Save results
            save_period=10,  # Save checkpoint every x epochs
            local_rank=-1,  # Local rank for distributed training
            exist_ok=True,  # Overwrite existing experiment
            pretrained=True,  # Use pretrained weights
            optimizer="auto",  # Optimizer (auto, SGD, Adam, etc.)
            verbose=True,  # Print verbose output
            seed=42,      # Random seed for reproducibility
            deterministic=True,  # Deterministic training
        )

        # Save the trained model
        model_path = Path("results/train/weights/best.pt")
        if model_path.exists():
            logger.info(f"Model saved successfully at {model_path}")
        else:
            logger.warning("Model file not found at expected location")

        return results

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 