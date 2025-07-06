# Universal Scanner Project Plan

## Overview
Build a universal scanner mobile app (React Native) that detects and decodes multiple code types — including barcodes, QR codes, license plates, container codes (horizontal/vertical), printed text, seals, Casio/LCD displays, and rail/airfreight identifiers — entirely offline and in real time.

## Development Phases

### Phase 1: Foundation & Demo App ✅ COMPLETED
**Duration**: Week 1-2  
**Status**: ✅ **COMPLETED**

- [x] Set up React Native project structure
- [x] Integrate `react-native-vision-camera` 
- [x] Create basic demo app with camera functionality
- [x] Set up development environment (Android/iOS)
- [x] Create initial project documentation

**Deliverables**:
- ✅ Working Expo camera demo in `/example/`
- ✅ Basic project structure and configuration
- ✅ Development environment setup

### Phase 2: Native C++ Core Development 🔄 IN PROGRESS
**Duration**: Week 3-4  
**Status**: 🔄 **IN PROGRESS**

- [ ] Design C++ core architecture with shared interfaces
- [ ] Implement MLKit integration bridges (Android JNI, iOS Obj-C++)
- [ ] Create ONNX Runtime integration for YOLOv8n detection
- [ ] Build frame processing pipeline in C++
- [ ] Implement domain-specific code processors

**Current Status**: Recently migrated to react-native-fast-tflite as baseline

### Phase 3: ONNX Runtime Migration 🔄 **NEXT**
**Duration**: Week 5-6  
**Status**: 🔄 **READY TO START**

**Performance Requirements**:
- Target 30 FPS (33ms per frame) processing time
- Handle 4K frames (~12MB each) efficiently using zero-copy patterns
- Maintain <1ms JSI overhead for native plugin calls

**Tasks**:
- [ ] **3.1**: Replace TensorFlow Lite with ONNX Runtime in C++ core
- [ ] **3.2**: Integrate YOLOv8n unified detection model from `/detection/models/`
- [ ] **3.3**: Implement proper ONNX nested array output handling (see ONNX-OUTPUT-FORMAT-DISCOVERY.md)
- [ ] **3.4**: Add MLKit integration for text/barcode recognition alongside ONNX
- [ ] **3.5**: Update VisionCamera Frame Processor plugin to call Universal Scanner
- [ ] **3.6**: Implement async processing with frame copying for complex operations

### Phase 4: Code Type Processors
**Duration**: Week 7-8  
**Status**: ⏳ **PENDING**

- [ ] Implement container code validation (ISO 6346)
- [ ] Build license plate processing logic
- [ ] Create seal code recognition pipeline
- [ ] Add support for LCD/Casio display reading
- [ ] Implement rail wagon and airfreight identifier processing

### Phase 5: Advanced Features
**Duration**: Week 9-10  
**Status**: ⏳ **PENDING**

- [ ] Manual targeting mode (tap to select)
- [ ] Multi-code detection in single frame
- [ ] Verbose debugging mode
- [ ] Configuration system (enable/disable code types)
- [ ] Regex filtering per code type

### Phase 6: Performance & Polish
**Duration**: Week 11-12  
**Status**: ⏳ **PENDING**

- [ ] Performance optimization and profiling
- [ ] Memory usage optimization
- [ ] Battery usage optimization
- [ ] Error handling and edge cases
- [ ] UI/UX improvements

### Phase 7: Plugin & Distribution
**Duration**: Week 13-14  
**Status**: ⏳ **PENDING**

- [ ] Package as `react-native-vision-camera` plugin
- [ ] Create comprehensive documentation
- [ ] Add example integration code
- [ ] Prepare for distribution
- [ ] Final testing and validation

## Key Architecture Decisions

### Core Architecture
- **UI Layer**: React Native (TypeScript) + `react-native-vision-camera`
- **Native Layer**: Shared C++ core (JNI for Android, Obj-C++ for iOS)
- **Detection**: YOLOv8n model (ONNX Runtime)
- **Recognition**: MLKit for proven reliability
- **Processing**: Domain-specific pipelines per code type

### Technical Stack
- **Frontend**: React Native, TypeScript, VisionCamera Frame Processors
- **JSI Bridge**: VisionCamera's proven Frame Processor Plugin architecture
- **Native**: C++, JNI (Android), Obj-C++ (iOS)
- **ML**: ONNX Runtime (YOLOv8n detection), MLKit (text/barcode recognition)
- **Build**: Gradle (Android), Xcode (iOS)
- **Performance**: Zero-copy frame processing, GPU acceleration, memory pools

## Current Status

**Overall Progress**: ~15% complete

**Recently Completed**:
- ✅ Successfully migrated to react-native-fast-tflite baseline
- ✅ Established working TensorFlow Lite example app with VisionCamera integration
- ✅ Documented critical ONNX output format discovery
- ✅ Recovered all Universal Scanner documentation and training assets
- ✅ Updated architecture with VisionCamera Frame Processor best practices

**Next Steps (Phase 3)**:
1. Test current TensorFlow Lite demo app on device
2. Replace TensorFlow Lite with ONNX Runtime in C++ core
3. Integrate YOLOv8n detection model with proper output handling
4. Add MLKit alongside ONNX for text/barcode recognition
5. Update plugin to call Universal Scanner instead of TensorFlow Lite

## Key Technical Discoveries

### ONNX Runtime React Native Output Format
⚠️ **Critical Discovery**: ONNX-RN returns nested arrays, NOT flat Float32Array!
- Documented in `ONNX-OUTPUT-FORMAT-DISCOVERY.md`
- Fixed confidence scores from 0.01 to 0.84+
- Essential for proper YOLO inference

### Development Principles
- **Stay Real**: Always use actual camera, never mock data
- **Plugin-First**: Build as reusable VisionCamera plugin
- **Performance-First**: Heavy processing in native C++ layer
- **Offline-First**: All inference runs on-device

## Risk Assessment

**High Risk**:
- [ ] ONNX Runtime performance on mobile devices
- [ ] Memory usage with multiple ML models loaded
- [ ] iOS build complexity with native dependencies

**Medium Risk**:
- [ ] MLKit integration complexity across platforms
- [ ] Model accuracy for diverse code types
- [ ] Battery usage optimization

**Low Risk**:
- [x] React Native foundation (mitigated - proven stack)
- [x] VisionCamera integration (mitigated - established plugin)

## Success Metrics

**Technical**:
- [ ] Real-time inference (<100ms per frame)
- [ ] >90% accuracy for supported code types
- [ ] <200MB memory usage during operation
- [ ] Support for both Android and iOS

**Business**:
- [ ] Successful integration into Cargosnap app
- [ ] Replacement of existing scanner modules
- [ ] Reusable plugin for other React Native apps