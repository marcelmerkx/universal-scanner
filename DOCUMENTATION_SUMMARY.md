# Documentation Summary

This document provides an overview of all documentation files in the Universal Scanner project.

## Core Documentation

### 1. **CLAUDE.md**
- **Purpose**: Project guidance for Claude Code AI assistant
- **Content**: Architecture overview, supported code types, DTOs, development approach, technical patterns
- **Key Info**: Contains critical ONNX output format discovery and MLKit integration patterns

### 2. **README.md**
- **Purpose**: Main project documentation for users and developers
- **Content**: Project overview, features, installation, usage examples, architecture diagram
- **Updated**: Replaced generic TFLite content with Universal Scanner specific documentation

### 3. **project_plan.md**
- **Purpose**: Development roadmap and progress tracking
- **Content**: 7 development phases, current status, technical decisions, success metrics
- **Status**: Phase 2 (Native C++ Core) in progress

### 4. **ONNX-OUTPUT-FORMAT-DISCOVERY.md**
- **Purpose**: Critical technical finding about ONNX Runtime React Native
- **Content**: Documents the nested array output format issue and solution
- **Impact**: Essential for anyone working with ONNX models in React Native

### 5. **MULTIPHASE_SCANNER_PLAN.md** (formerly multiphase_scanner.md)
- **Purpose**: Phased implementation plan and technical notes
- **Content**: Code type priorities, ONNX output format details, implementation references
- **Note**: Contains duplicate ONNX info that's better organized in dedicated file

### 6. **CONTRIBUTING.md**
- **Purpose**: Guidelines for contributors
- **Content**: Development setup, workflow, project structure, contribution areas
- **Updated**: Customized for Universal Scanner from generic template

### 7. **example/README.md**
- **Purpose**: Documentation for the demo application
- **Content**: Setup instructions, features, configuration, troubleshooting
- **Updated**: Replaced generic React Native content with scanner-specific guide

## Key Technical Documentation Points

1. **ONNX Output Format**: Critical discovery that ONNX-RN returns nested arrays, not flat Float32Array
2. **Architecture**: Hybrid approach using YOLO for detection + MLKit for recognition
3. **Supported Types**: 11 different code types with domain-specific validation
4. **Development Phases**: Currently in Phase 2 of 7, building native C++ core

## Documentation Maintenance

- Update `project_plan.md` after each development milestone
- Keep `CLAUDE.md` current with architectural decisions
- Document new code types in both README.md and CLAUDE.md
- Add technical discoveries to appropriate technical docs

## Missing Documentation (To Create)

1. **API.md**: Detailed API reference for all public methods
2. **TROUBLESHOOTING.md**: Common issues and solutions
3. **PERFORMANCE.md**: Optimization guide and benchmarks
4. **MODELS.md**: ML model training and integration guide