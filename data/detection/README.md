# Detection Model Conversion

## Model Format Decision

**Selected Format: ONNX**
- File: `models/roboflow/unified_detection_v7.onnx`
- Runtime: ONNX Runtime Mobile

## Why ONNX Over TFLite

### Direct Conversion Issues
- **PyTorch → TFLite**: No direct conversion path exists
- **Multi-step conversion**: PyTorch → ONNX → TensorFlow → TFLite introduces complexity and potential errors
- **Dependency conflicts**: TensorFlow installation often fails on certain Python versions

### ONNX Advantages
- **Single conversion step**: PyTorch → ONNX (clean, well-supported)
- **Cross-platform**: Works identically on iOS and Android
- **Performance**: ONNX Runtime Mobile is highly optimized
- **Established**: Already working in codebase with proper output format handling

### Performance Comparison
- **ONNX Runtime Mobile**: ~10-20ms inference on mobile devices
- **TFLite**: Similar performance but with conversion complexity
- **PyTorch Mobile**: Larger model size, slower cold start

## Model Conversion Scripts

### Available Scripts
- `scripts/convert_pt_to_tflite.py` - Full PyTorch to TFLite pipeline (complex)
- `scripts/convert_onnx_to_tflite.py` - ONNX to TFLite with fallbacks
- `scripts/convert_to_onnx.py` - Simple PyTorch to ONNX (recommended)

### Recommended Workflow
```bash
# Convert PyTorch model to ONNX (if needed)
python scripts/convert_to_onnx.py

# Use existing ONNX model directly
# File: models/roboflow/unified_detection_v7.onnx
```

## Integration Notes

### Output Format Handling
Critical discovery: ONNX-RN returns nested arrays, not flat Float32Array.
See `/ONNX-OUTPUT-FORMAT-DISCOVERY.md` for implementation details.

### Mobile Deployment
- **Android**: ONNX Runtime via JNI bridge
- **iOS**: ONNX Runtime via C++ integration
- **React Native**: Plugin architecture with native processing

## Model Specifications
- **Input**: [1, 3, 640, 640] Float32
- **Output**: [1, 9, 8400] or [1, 8400, 9] (adaptive indexing required)
- **Classes**: 5 detection classes + 4 bbox coordinates
- **Size**: ~12MB ONNX vs ~8MB TFLite (trade-off for simplicity)