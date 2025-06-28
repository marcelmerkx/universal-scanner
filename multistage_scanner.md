# Multistage Scanner Implementation Approaches

This document outlines the evolution of our thinking about implementing the universal scanner, documenting failed approaches and leading to our final chosen strategy.

## Approach Evolution

### ~~Approach 1: Full Custom Implementation~~ ❌ ABANDONED
**Date**: Early December 2024  
**Description**: Build everything from scratch with custom ML pipeline  
**Why Abandoned**: Too complex, reinventing the wheel, high risk of poor performance

### ~~Approach 2: Pure MLKit Integration~~ ❌ ABANDONED  
**Date**: Mid December 2024  
**Description**: Use only Google MLKit for all detection and recognition  
**Why Abandoned**: Limited to supported formats, no custom object detection for containers/seals

### ~~Approach 3: ONNX-First Architecture~~ ❌ ABANDONED
**Date**: Late December 2024  
**Description**: Use ONNX Runtime for both detection and recognition  
**Why Abandoned**: Poor text recognition compared to MLKit, integration complexity

### ~~Approach 4: Multiple Plugin Integration~~ ❌ ABANDONED
**Date**: Early January 2025  
**Description**: Combine multiple existing React Native scanner libraries  
**Why Abandoned**: Inconsistent APIs, performance overhead, maintenance burden

## ✅ Final Chosen Approach: Forking Strategy (2025-06-28)

### Strategy: Fork react-native-fast-tflite and Migrate to ONNX
**Decision Date**: 2025-06-28  
**Status**: ✅ **ADOPTED**

**Rationale**:
1. **Proven Foundation**: react-native-fast-tflite has working VisionCamera integration and JSI bindings
2. **Solid Architecture**: Established C++ core with proper native module setup
3. **Migration Path**: Can systematically replace TensorFlow Lite with ONNX Runtime
4. **Risk Mitigation**: Start with known-working baseline, then incrementally improve

**Implementation Plan**:
1. ✅ **Phase 1**: Clone react-native-fast-tflite as baseline
2. ✅ **Phase 2**: Ensure TensorFlow Lite example works correctly
3. 🔄 **Phase 3**: Replace TensorFlow Lite with ONNX Runtime 
4. ⏳ **Phase 4**: Add MLKit integration for text recognition
5. ⏳ **Phase 5**: Implement multi-model detection pipeline
6. ⏳ **Phase 6**: Add domain-specific code processors
7. ⏳ **Phase 7**: Package as Universal Scanner plugin

**Key Benefits**:
- ✅ Proven VisionCamera Frame Processor integration
- ✅ Working JSI bindings and native module architecture  
- ✅ Cross-platform C++ foundation (Android/iOS)
- ✅ Established build system and dependency management
- ✅ Known performance characteristics

**Migration Strategy**:
- Keep existing API surface where possible
- Replace TensorFlow Lite with ONNX Runtime incrementally
- Add MLKit as additional recognition engine
- Extend with Universal Scanner specific features

## Key Technical Decisions

### Native Architecture
- **Language**: C++ for shared core logic
- **Platform Integration**: JNI (Android) + Obj-C++ (iOS)
- **ML Runtime**: ONNX Runtime (detection) + MLKit (recognition)
- **Framework**: react-native-vision-camera Frame Processors

### Performance Strategy
- **Heavy Processing**: Keep in native C++ layer
- **Bridge Usage**: Minimize React Native bridge calls
- **Memory Management**: Efficient tensor handling and cleanup
- **Threading**: Separate ML inference from UI thread

### Model Pipeline
1. **Frame Capture**: VisionCamera provides camera frames
2. **Object Detection**: ONNX Runtime + YOLOv8n model
3. **Text Recognition**: MLKit for detected regions
4. **Post-Processing**: Domain-specific validation and formatting
5. **Result Delivery**: Structured ScanResult objects to React Native

## Lessons Learned

1. **Don't Reinvent**: Leverage proven libraries like MLKit for text recognition
2. **Start Simple**: Begin with working foundation, then extend
3. **Performance First**: Native implementation crucial for real-time scanning
4. **Plugin Architecture**: VisionCamera Frame Processors provide optimal integration
5. **Incremental Migration**: Systematic replacement reduces risk

## Decision Rationale

The forking strategy provides the best balance of:
- **Low Risk**: Starting with proven, working codebase
- **High Performance**: Native C++ core with JSI integration
- **Extensibility**: Can add Universal Scanner features incrementally
- **Maintainability**: Building on established architecture patterns

This approach allows us to focus on the Universal Scanner specific logic (multi-model detection, domain processing) rather than solving basic infrastructure problems.