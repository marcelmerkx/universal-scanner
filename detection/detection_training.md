# Training ONNX "DETECTION" (phase 2) model

## Background
we have decided on a YOLO (like a stable v8n) ONNX object recognition approach for our "detection" phase. 
This requires us to run the ONNX runtime using an .onnx file trained to recognise each of our different categories/classes. 

## Tools to use:
* Python scripts (see some in /legacy-notes already) (for visualising issues and troubleshooting we may pop over to Jupyter notebooks, as well as the Colab part, but frankly, I do prefer pure python on the command line for the bulk stuff)
* Roboflow. In our .env in the root of the document there is a ROBOFLOW_API_KEY to use but to never parse or check in (!!).
* ultralytics for YOLO8 training and conversion to ONNX
* in the folder models/legacy-notes I placed notes and scripts from our earlier succesful prototype for vertical container scanning. Great to load into context and cherry pick!

## our baseline data
* the good news: we have images. Loads of images, and actually also with expected (mostly correct) codes. What I imagine we could do is make a folder structure such as "detection/training_data/00_raw/" and in there make folders (see detection/training_data for empty folder structure as proposed) and name the file the expected code output. That data will initially be a bit dirty (not all labels correct, sometimes a _v and _h container will show up in the wrong folder etc), but probably ~90% right
* the sad news: no bounding boxes
* Roboflow has a 'universe' and I spotted nice open source (attribution) datasets with labels in probably most of our classes (like container codes, license plates, etc). This means we could (if we think that works) try to make small pre-labelling models that take our individual datasets, and train a small (quick and dirty) model to get the bounding boxes for us on our own images (for which we know the outcomes, which we do not have in the universe-sourced images) and add them to the (same?) folder (yolo8 file format).

## Pipeline

### Overview: Iterative Bootstrapping Approach
We will minimize manual labeling through a multi-stage bootstrapping process:

1. **Stage 1 - Foundation Models**: Use Roboflow Universe open datasets to train basic detection models for each code type
2. **Stage 2 - Auto-Labeling**: Apply foundation models to our raw images to generate initial bounding boxes
3. **Stage 3 - Quality Control**: Automatically filter high-confidence predictions and validate against known codes
4. **Stage 4 - Synthetic Augmentation**: Generate synthetic variations to expand dataset
5. **Stage 5 - Iterative Refinement**: Train improved models and repeat auto-labeling with better accuracy
6. **Stage 6 - Final Model**: Combine all code types into unified multi-class YOLOv8n model

### Key Strategies to Minimize Manual Labeling:
- **Transfer Learning**: Start with pre-trained models from Roboflow Universe
- **Confidence Filtering**: Only use high-confidence auto-labeled data (>0.85 confidence)
- **Cross-Validation**: Use filename codes to validate detection accuracy
- **Scale Real Data**: Process thousands of real images automatically rather than synthetic augmentation
- **Active Learning**: Identify and manually label only the most informative samples
- **Progressive Refinement**: Each iteration improves auto-labeling accuracy

## Plan

### Phase 1: Data Organization & Foundation Setup (Week 1)

#### 1.1 Organize Raw Data
**Script**: `organize_raw_images.py`
- Scan existing image archives for each code type
- Extract expected codes from filenames/metadata
- Copy to appropriate folders in `00_raw/` structure
- Generate inventory report: `raw_data_inventory.json`
- Flag potentially mislabeled images based on filename patterns

**Script:**: `relabel_container_images.py`
Specifically for container codes:
- we trained a model recognising single and double line horizontal codes as well as vertical codes. This script will then re-label the YOLOs based on the bounding box sizes (tall means vertical, narrow means horizontal and a ratio kind of in between means 2 lines :)

#### 1.2 Download Roboflow Universe Datasets
**Script**: `download_universe_datasets.py`
- Search and download relevant open-source datasets:
  - License plates: "license-plate-detection" datasets
  - QR codes: "qr-code-detection" datasets
  - Container codes: "shipping-container" datasets
  - Barcodes: "barcode-detection" datasets
- Store in `detection/universe_datasets/` with attribution
- Create mapping file: `universe_datasets_catalog.json`

#### 1.3 Train Foundation Models
**Script**: `train_foundation_models.py`
- For each code type, train a basic YOLOv8n model using Universe data
- Use transfer learning from YOLOv8n pretrained weights
- Quick training: 50-100 epochs, early stopping
- Save models in `detection/models/foundation/`
- Generate performance metrics: `foundation_models_report.md`

### Phase 2: Auto-Labeling Pipeline (Week 2)

#### 2.1 Initial Auto-Labeling
**Script**: `auto_label_batch.py`
- Process images from `00_raw/` through foundation models
- For each image:
  - Run appropriate foundation model
  - Filter detections by confidence (>0.7 for initial pass)
  - Save YOLO format labels
  - Track confidence scores and detection metadata
- Output to `01_ready_to_upload/auto_labeled/`

#### 2.2 Validation & Quality Control
**Script**: `validate_auto_labels.py`
- Cross-check detected text against expected codes from filenames
- For container codes: validate ISO 6346 checksums
- For license plates: validate format patterns
- Generate quality report: `auto_label_quality_report.json`
- Move high-confidence validated samples to `01_ready_to_upload/verified/`

#### 2.3 Handle Edge Cases
**Script**: `identify_edge_cases.py`
- Identify images with:
  - No detections
  - Low confidence detections (<0.7)
  - Multiple conflicting detections
  - Validation failures
- Generate priority list for manual review: `edge_cases_priority.csv`
- Create visual inspection notebook: `inspect_edge_cases.ipynb`

### Phase 3: Scale Real Data Processing (Week 2-3)

#### 3.1 Expand Real Data Collection
**Script**: `expand_data_sources.py`
- Scan additional image archives and sources
- Process historical logistics data
- Include edge cases: damaged, worn, angled codes
- Target: 2,000-5,000 real images per class

#### 3.2 Batch Auto-Labeling at Scale
**Script**: `scale_auto_labeling.py`
- Process larger batches efficiently
- Use GPU acceleration for faster inference
- Parallel processing across multiple foundation models
- Focus on volume over synthetic variations

#### 3.3 Quality-Focused Hard Negatives
**Script**: `collect_real_negatives.py`
- Find real images with no valid codes (empty containers, damaged labels)
- Use actual confusing scenarios from logistics
- More realistic than synthetic hard negatives

### Phase 4: Roboflow Integration (Week 3)

#### 4.1 Prepare Upload Batches
**Script**: `prepare_roboflow_upload.py`
- Organize data into balanced train/val/test splits (70/20/10)
- Ensure class balance across splits
- Create upload manifest: `upload_manifest.json`
- Package in Roboflow-compatible format

#### 4.2 Upload to Roboflow
**Script**: `upload_to_roboflow.py`
- Use Roboflow API to create project
- Upload images and annotations in batches
- Apply Roboflow augmentations:
  - Additional rotations
  - Cutout augmentation
  - Mosaic augmentation
- Generate and download augmented dataset

#### 4.3 Manual Review Session
**Manual Process**: 
- Review auto-labeled data in Roboflow interface
- Focus on edge cases identified earlier
- Correct obvious errors
- Add missing annotations for high-value samples
- Time budget: 4-6 hours maximum

### Phase 5: Iterative Model Training (Week 3-4)

#### 5.1 Train Improved Models
**Script**: `train_iterative_models.py`
- Train new YOLOv8n models with expanded dataset
- Use different model sizes for experimentation:
  - YOLOv8n: Fastest, for real-time
  - YOLOv8s: Balanced
  - YOLOv8m: Higher accuracy (for auto-labeling)
- Track metrics: mAP, precision, recall per class

#### 5.2 Second-Pass Auto-Labeling
**Script**: `auto_label_round2.py`
- Use improved models on remaining unlabeled data
- Higher confidence threshold (>0.85)
- Focus on previously failed images
- Validate and add to dataset

#### 5.3 Active Learning Selection
**Script**: `select_active_learning_samples.py`
- Identify most informative samples for manual labeling:
  - High model uncertainty (conflicting predictions)
  - Edge of decision boundary
  - Rare/unique visual patterns
- Generate ranked list: `active_learning_priority.csv`
- Limit to 100-200 images for manual review

### Phase 6: Final Model Development (Week 4-5)

#### 6.1 Combine All Classes
**Script**: `prepare_unified_dataset.py`
- Merge all verified labeled data across code types
- Balance classes to prevent bias
- Create final train/val/test splits
- Generate class mapping: `unified_classes.yaml`

#### 6.2 Train Production Model
**Script**: `train_production_model.py`
- Train final YOLOv8n model with all classes
- Optimize hyperparameters:
  - Learning rate scheduling
  - Augmentation parameters
  - Loss weights
- Target metrics:
  - mAP@0.5: >0.85
  - Inference time: <30ms on mobile GPU

#### 6.3 Model Optimization
**Script**: `optimize_and_export_model.py`
- Quantize model to INT8 for mobile deployment
- Export to ONNX format
- Validate accuracy loss from quantization (<5% mAP drop)
- Generate model card: `model_documentation.md`

### Phase 7: Validation & Testing (Week 5)

#### 7.1 Comprehensive Testing
**Script**: `test_production_model.py`
- Test on held-out test set
- Test on real-world "dirty" images
- Measure performance per class
- Edge case testing (blur, occlusion, angles)

#### 7.2 Performance Profiling
**Script**: `profile_model_performance.py`
- Measure inference speed on target devices
- Memory usage analysis
- Battery consumption estimates
- Generate performance report: `performance_benchmarks.md`

#### 7.3 Create Demo
**Script**: `create_detection_demo.py`
- Build interactive demo showing detections
- Include confidence scores and class labels
- Visual comparison with ground truth

### Monitoring & Iteration

#### Continuous Improvement Loop
**Script**: `monitor_and_improve.py`
- Track model performance in production
- Collect failure cases
- Periodically retrain with new data
- A/B test model versions

### Success Metrics
- **Manual labeling time**: <8 hours total
- **Dataset size**: 2,000-5,000 real labeled images per class (minimal augmentation needed)
- **Model mAP**: >0.80 across all classes
- **Inference speed**: <30ms on Snapdragon 660
- **False positive rate**: <5%

### Risk Mitigation
- **Poor auto-labeling accuracy**: Use ensemble of models for consensus
- **Class imbalance**: Synthetic data generation and class weights
- **Domain shift**: Collect diverse data sources
- **Annotation noise**: Multiple validation passes and confidence filtering