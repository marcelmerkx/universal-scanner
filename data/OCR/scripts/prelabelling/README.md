# YOLOv8 Container OCR Auto-labeling Script

A comprehensive Python script for automatically pre-labeling container images using an existing YOLOv8 vertical OCR model. This tool is designed to accelerate the creation of training datasets for container character recognition by leveraging your existing trained model to generate initial labels.

## 🚀 Features

- **Batch Processing**: Efficiently processes thousands of images in batches
- **Vertical Container Optimization**: Specifically designed for vertical container codes with constraints
- **Quality Control**: Advanced filtering to ensure high-quality labels
- **Confidence Thresholding**: Configurable confidence levels for detection filtering
- **Box Constraints**: Limits boxes to maximum 11 per image with size similarity checks
- **Overlap Removal**: Non-Maximum Suppression to eliminate redundant detections
- **YOLO Format Output**: Generates annotations ready for YOLOv8 training
- **Comprehensive Logging**: Detailed statistics and progress tracking
- **Command-line Interface**: Easy-to-use CLI with customizable parameters

## 📋 Requirements

- Python 3.7+
- YOLOv8 (ultralytics)
- PyTorch
- OpenCV
- NumPy
- tqdm

## 🔧 Installation

1. Clone or download the script files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Usage

```bash
python container_ocr_auto_labeler.py --model your_vertical_model.pt --input /path/to/images --output /path/to/labels
```

### Advanced Usage

```bash
python container_ocr_auto_labeler.py \
    --model vertical_model.pt \
    --input training_images/ \
    --output labels/ \
    --confidence 0.4 \
    --max-boxes 10 \
    --min-area 150 \
    --batch-size 64 \
    --extensions .jpg .png .tiff
```

### Parameters

- `--model`: Path to your trained YOLOv8 model (.pt file) **[Required]**
- `--input`: Directory containing images to label **[Required]**
- `--output`: Directory to save YOLO format annotations **[Required]**
- `--confidence`: Confidence threshold (default: 0.3)
- `--max-boxes`: Maximum boxes per image (default: 11)
- `--min-area`: Minimum box area in pixels (default: 100)
- `--batch-size`: Batch size for processing (default: 32)
- `--extensions`: Image file extensions to process (default: .jpg .jpeg .png)

## 🔍 How It Works

### 1. **Image Processing**
- Loads images in batches for efficient processing
- Runs inference using our existing vertical OCR model
- Applies initial confidence filtering

### 2. **Vertical Container Constraints**
- **Size Similarity**: Removes boxes that are significantly different in size from the median
- **Maximum Boxes**: Limits to 11 boxes per image (configurable)
- **Vertical Sorting**: Arranges boxes from top to bottom for vertical containers
- **Overlap Removal**: Uses Non-Maximum Suppression to eliminate redundant detections

### 3. **Quality Control**
- **Confidence Filtering**: Only keeps detections above specified threshold
- **Area Filtering**: Removes boxes smaller than minimum area
- **IoU-based NMS**: Removes overlapping boxes with lower confidence
- **Size Consistency**: Ensures character boxes are similar in size

### 4. **Output Generation**
- Converts bounding boxes to YOLO format (normalized coordinates)
- Saves annotations as `.txt` files matching image names
- Generates processing statistics and logs

## 📊 Output Structure

```
output_directory/
├── image1.txt          # YOLO format annotations
├── image2.txt
├── ...
├── auto_labeling.log   # Processing log
└── labeling_stats.json # Statistics summary
```

### YOLO Format
Each annotation file contains lines in the format:
```
class_id x_center y_center width height
```
Where all coordinates are normalized (0-1 range).

## 📈 Statistics Tracking

The script tracks and reports:
- Total images processed
- Success/failure rates
- Total boxes generated
- Average confidence scores
- Processing time
- Quality control metrics

## 🛠️ Customization

### For Different Container Types
- Adjust `--max-boxes` for containers with different character counts
- Modify `--confidence` threshold based on your model's performance
- Change `--min-area` for different image resolutions

### For Quality Control
- Increase confidence threshold for higher precision
- Adjust `min_area` to filter out noise
- Modify size variation factor in the code for stricter size matching

## 🎯 Best Practices

1. **Start with Conservative Settings**: Begin with higher confidence thresholds and adjust downward
2. **Validate Results**: Always review a sample of generated labels before using for training
3. **Batch Size**: Adjust batch size based on your GPU memory
4. **Backup Original Data**: Keep copies of your original images
5. **Monitor Logs**: Check processing logs for any issues or patterns

## 🚨 Important Notes

- This script is designed for **vertical container codes** where characters are arranged top-to-bottom
- Generated labels are **pre-labels** and should be reviewed before training
- The script assumes your vertical model is already well-trained and producing good results
- For best results, ensure your input images are similar to your original training data

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure your model path is correct and the model is compatible
2. **No Detections**: Lower the confidence threshold or check image quality
3. **Too Many Boxes**: Increase confidence threshold or reduce max-boxes parameter
4. **Memory Issues**: Reduce batch size or process smaller image batches

### Performance Tips

- Use GPU acceleration if available
- Process images in smaller batches if memory constrained
- Consider resizing very large images for faster processing
- Monitor system resources during processing

## 📝 Example Workflow

1. **Prepare your data**:
   ```bash
   mkdir training_images labels
   # Copy your images to training_images/
   ```

2. **Run auto-labeling**:
   ```bash
   python container_ocr_auto_labeler.py --model vertical_model.pt --input training_images/ --output labels/
   ```

3. **Review results**:
   - Check `auto_labeling.log` for processing summary
   - Review `labeling_stats.json` for statistics
   - Spot-check some annotation files

4. **Use for training**:
   ```bash
   # Your labels are now ready for YOLOv8 training
   yolo detect train data=your_dataset.yaml model=yolov8n.pt epochs=100
   ```

## 📞 Support

This script is designed to be robust and handle various edge cases, but if you encounter issues:

1. Check the log file (`auto_labeling.log`) for detailed error messages
2. Verify your model and input paths are correct
3. Ensure your images are in supported formats
4. Try with a smaller batch size if experiencing memory issues

## 🎉 Success Metrics

A successful run typically shows:
- Success rate > 90%
- Average confidence > 0.4
- 3-11 boxes per image for container codes
- Consistent processing speed across batches

This tool should significantly speed up your labeling process while maintaining quality standards for your next YOLOv8 training run!