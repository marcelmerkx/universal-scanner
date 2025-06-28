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

### Phase 3: ML Model Integration
**Duration**: Week 5-6  
**Status**: ⏳ **PENDING**

- [ ] Integrate YOLOv8n ONNX model for object detection
- [ ] Configure MLKit for text recognition and barcode scanning
- [ ] Implement model switching and fallback logic
- [ ] Optimize inference performance for mobile devices
- [ ] Add GPU delegate support where available

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
- **Frontend**: React Native, TypeScript, VisionCamera
- **Native**: C++, JNI (Android), Obj-C++ (iOS)
- **ML**: MLKit, ONNX Runtime, YOLOv8n
- **Build**: Gradle (Android), Xcode (iOS)

## Current Status

**Overall Progress**: ~15% complete

**Recently Completed**:
- ✅ Successfully migrated to react-native-fast-tflite baseline
- ✅ Established working TensorFlow Lite example app
- ✅ Documented critical ONNX output format discovery

**Next Steps**:
1. Migrate from TensorFlow Lite to ONNX Runtime
2. Implement universal scanner plugin architecture
3. Begin C++ core development with MLKit integration

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