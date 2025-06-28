# Container Digit Detection System

A comprehensive computer vision system for detecting and reading container numbers from shipping containers. This repository contains both the **ML/Python training pipeline** and **mobile applications** for real-time container scanning.

## 🏗️ Project Structure

```
container-digit-detector/           # 🐍 ML/Python Workspace
├── models/                        # ONNX models and training artifacts
├── data/                         # Training datasets and validation images
├── notebooks/                    # Jupyter notebooks for development
├── scripts/                      # Python training and processing scripts
├── config/                       # Configuration files
├── requirements.txt              # Python dependencies
├── *.md                         # Documentation and guides
└── apps/                        # 📱 Mobile Applications
    ├── ContainerCameraApp/       # Modern ONNX-powered camera app
    └── ContainerDigitDetector/   # Original reference implementation
```

## 🎯 Components

### 🐍 ML/Python Pipeline
**Purpose**: Model training, data processing, and ONNX export for mobile deployment.

- **YOLO Training**: Character detection model training
- **ONNX Export**: Mobile-optimized model conversion
- **Data Processing**: Dataset preparation and validation
- **Jupyter Notebooks**: Development and experimentation

### 📱 Mobile Applications
**Purpose**: Real-time container scanning on mobile devices.

- **ContainerCameraApp**: Modern camera app with ONNX inference
- **ContainerDigitDetector**: Original full-featured implementation

## 🚀 Quick Start

### For ML Development
```bash
# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train models or run notebooks
python scripts/train_model.py
jupyter notebook notebooks/
```

### For Mobile Development
```bash
# Navigate to the camera app
cd apps/ContainerCameraApp
npm install
npx expo run:android  # or npx expo run:ios
```

## 📊 Performance

- **OCR Accuracy**: **100%** on validation set (15 images)
- **ONNX Inference**: ~20-30ms per image on mobile devices
- **Character Detection**: Supports 11-12 character container numbers
- **Real-time Processing**: Vertical overlay guide for optimal positioning

## 🔧 Technical Features

### ML Pipeline
- **YOLO Character Detection**: Individual character bounding boxes
- **ONNX Runtime**: Mobile-optimized inference
- **Two-Stage Filtering**: Eliminates overlapping/misaligned detections
- **Container Code Corrections**: Smart character replacement (O↔0, I↔1, etc.)

### Mobile Apps
- **Real-time Camera**: Live preview with scanning overlay
- **Vertical Guide**: Optimized for container number positioning
- **Error Handling**: Explicit failures without fallback solutions
- **Professional UI**: Modern interface with loading and error states

## 📱 Mobile App Details

### ContainerCameraApp (Recommended)
- **Status**: ✅ Active development
- **Tech**: React Native 0.79.4, Expo SDK 53, ONNX Runtime
- **Features**: Clean architecture, real ONNX inference, vertical scanning

### ContainerDigitDetector (Reference)
- **Status**: 📚 Archive/Reference
- **Tech**: React Native, Expo, Native modules
- **Features**: Full pipeline, image stitching, comprehensive validation

## 📋 Development Workflow

1. **ML Development**: Train and validate models in Python
2. **ONNX Export**: Convert models for mobile deployment
3. **Mobile Integration**: Test models in ContainerCameraApp
4. **Production**: Deploy mobile app with optimized models

## 🔗 Key Documentation

- [`apps/README.md`](apps/README.md) - Mobile app development guide
- [`RN-Vision-Camera.md`](RN-Vision-Camera.md) - Technical implementation plan
- [`ONNX-README.md`](ONNX-README.md) - ONNX model integration guide
- [`requirements.md`](requirements.md) - Detailed system requirements

## 🎯 Next Steps

### Production Readiness
1. **Expand Dataset**: Collect 1K+ annotated container images
2. **Model Optimization**: Quantization and pruning for mobile
3. **Native Frame Processing**: Kotlin/Swift frame processors
4. **CI/CD Pipeline**: Automated testing and deployment

### Mobile Features
1. **Offline Mode**: Complete on-device processing
2. **Batch Processing**: Multiple container scanning
3. **Data Sync**: Cloud synchronization for results
4. **Analytics**: Performance metrics and usage tracking

## 🤝 Contributing

1. **ML Development**: Use the Python pipeline for model improvements
2. **Mobile Development**: Focus on ContainerCameraApp for new features
3. **Documentation**: Update relevant `.md` files for changes
4. **Testing**: Validate changes on both training and mobile sides

## 📄 License

This project is developed for container digit detection research and development. 