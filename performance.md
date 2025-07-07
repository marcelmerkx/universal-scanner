# Mobile Object Detection Performance Optimization

This document outlines our performance optimization strategy for real-time object detection on mobile devices, based on extensive testing and research.

**Key Finding**: For small YOLO models on mobile devices, **model size optimization significantly outperforms hardware acceleration** due to GPU delegation overhead and setup costs.

---

## 🧠 Goals

- Achieve optimal real-time performance (15+ FPS) on mobile devices
- Minimize latency while maintaining detection accuracy
- Use proven optimization techniques over experimental hardware acceleration
- Provide scalable performance across different device capabilities

---

## 🔍 Real-World Testing Results (Samsung S24)

### GPU Acceleration Analysis
- **NNAPI Testing**: Attempted hardware acceleration via NNAPI on Samsung S24
- **Result**: No performance improvement, likely due to:
  - GPU delegation overhead exceeding computation time for small models
  - NNAPI falling back to CPU execution with additional overhead
  - Memory transfer costs between CPU and GPU
  - ARM Mali GPU limitations with ONNX Runtime

### Model Size Optimization Success
- **320x320 Model**: 20 FPS (43% faster than 640x640)
- **416x416 Model**: 17 FPS (21% faster than 640x640)  
- **640x640 Model**: 14 FPS (baseline)
- **Accuracy Impact**: Only 1% confidence drop (70% → 69%)

**Conclusion**: Model size reduction delivers real performance gains with minimal accuracy loss.

---

## 🎯 Adopted Strategy: Model Size Optimization

Instead of pursuing GPU acceleration, we optimize through:

### 1. **Flexible Model Sizes**
- **320x320**: Best for real-time applications (20 FPS)
- **416x416**: Balanced performance/quality (17 FPS)  
- **640x640**: Highest accuracy when FPS is not critical (14 FPS)

### 2. **Dynamic Model Selection**
Users can switch between model sizes based on use case:
- **High-speed scanning**: 320x320
- **Balanced operation**: 416x416
- **Maximum accuracy**: 640x640

---

### 1. **Model Quantization** (Recommended Next Step)
- **INT8 Quantization**: Reduce model size by ~75% with minimal accuracy loss
- **FP16 Quantization**: 50% size reduction, better accuracy retention than INT8
- **Expected Gain**: Additional 20-30% performance improvement

### 2. **Preprocessing Optimizations**
- **YUV Resizing**: Resize before RGB conversion (already implemented)
- **NEON SIMD**: ARM optimization for image processing operations
- **Memory Pooling**: Reuse allocated buffers to reduce allocation overhead

### 3. **Threading Optimizations**
- **Optimal Thread Count**: 4 threads for modern mobile CPUs (already implemented)
- **Frame Dropping**: Skip processing if previous frame still processing
- **Async Processing**: Process frames on background thread

### 4. **Model Architecture Improvements**
- **Pruning**: Remove less important neurons/weights
- **Knowledge Distillation**: Train smaller student model from larger teacher
- **MobileNet Backbone**: Switch to mobile-optimized backbone architecture

### 5. **Platform-Specific Optimizations**

#### Android
- **ARM NEON**: Vectorized CPU operations
- **CPU Governor**: Request performance mode during scanning
- **Thermal Throttling**: Monitor and adapt to device thermal state

#### iOS  
- **Accelerate Framework**: Apple's optimized linear algebra
- **Core ML**: For iOS-specific deployments
- **Metal Performance Shaders**: GPU compute if beneficial

---

## 🚀 Performance Roadmap

### Immediate (Implemented)
- ✅ Flexible model sizes (320/416/640)
- ✅ Optimized preprocessing pipeline
- ✅ Multi-threaded CPU execution

### Short Term (Next Sprint)
- 🎯 **INT8 Quantization**: Expected 25% additional speedup
- 🎯 **Memory Pool Optimization**: Reduce allocation overhead
- 🎯 **Frame Rate Limiting**: Prevent unnecessary processing

### Medium Term
- 📋 **Model Pruning**: Further size reduction while maintaining accuracy
- 📋 **Custom Mobile Architecture**: YOLOv8n-mobile variant
- 📋 **Benchmark Suite**: Automated performance testing across devices

### Research
- 🔬 **Knowledge Distillation**: Ultra-lightweight models (Sub-320x320)
- 🔬 **Dynamic Resolution**: Adaptive quality based on detection confidence
- 🔬 **Edge TPU**: Google Coral integration for specific deployments

---

## 📊 Current Performance Baseline

| Model Size | FPS | Accuracy | Use Case |
|------------|-----|----------|----------|
| 320x320 | 20 | 69% | Real-time scanning |
| 416x416 | 17 | 69-70% | Balanced operation |
| 640x640 | 14 | 70% | Maximum accuracy |

**Target**: Achieve 30+ FPS with 320x320 + INT8 quantization while maintaining 65%+ accuracy.

---