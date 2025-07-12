# Configuration Template for Container OCR Auto-labeling
# Copy this file and modify the values as needed

# Model Configuration
MODEL_PATH = "ContainerCameraApp/assets/models/best-OCR-Colab-22-06-25.pt"

# Data Configuration
INPUT_DIR = "data/OCR/cutouts"
OUTPUT_DIR = "data/OCR/labels"

# Processing Parameters
CONFIDENCE_THRESHOLD = 0.3      # Minimum confidence for detections
MAX_BOXES_PER_IMAGE = 11        # Maximum boxes per image
MIN_BOX_AREA = 100.0           # Minimum box area in pixels
BATCH_SIZE = 32                # Images to process in each batch

# Quality Control Parameters
IOU_THRESHOLD = 0.5            # IoU threshold for overlap removal
SIZE_VARIATION_FACTOR = 3.0    # Maximum size variation from median

# Supported file extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Logging Configuration
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR
SAVE_STATISTICS = True         # Save processing statistics
