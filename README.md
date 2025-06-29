# Universal Scanner

A high-performance React Native plugin for real-time detection and decoding of multiple code types using computer vision and machine learning.

## Overview

Universal Scanner is a `react-native-vision-camera` plugin that combines YOLO object detection with MLKit text recognition to scan and decode various types of codes and text in real-time. Built with a hybrid native architecture for optimal performance.

We currently have a **working vertical container code scanner**, including documentation and source code. This will be included in a dedicated (possibly temporary) subfolder within this project.

**✅ DEMO APP READY**: A working Expo camera demo is available in `/example/` - see the README there for setup instructions.

---

## 🎯 Functional Scope

### Core Capabilities
- Real-time scanning via camera (video stream)
- Offline model inference using YOLOv8n (ONNX) + OCR
- Dynamic toggle to enable/disable scanner types
- Manual targeting mode: user selects code from detected bounding boxes
- Cross-platform UI via React Native
- Native inference pipelines in C++ (shared core) bridged to Android (JNI) and iOS (Obj-C++)
- Output: DTOs with decoded value, class, bbox, image references, and metadata
- Input: DTO for scanning configuration (e.g. enabled modes, regex filters, manual mode)
- Built as a reusable `react-native-vision-camera` plugin for drop-in use
- Demo app included for local development and testing
- Performant: considering we need this to be a high-performing solution where we will do all complex processing on the native end and avoid significant use of the Native <--> RN bridge which we know to be really slow

---

## 👤 User Stories

### US001 - Scan Anything Instantly
**As a** logistics worker,  
**I want** to point my phone at any code (QR, barcode, plate, container ID),  
**so that** I get a near instant and accurate result without switching modes.

### US002 - Scanner Type Toggle
**As a** field operator,  
**I want** to disable certain types of scans,  
**so that** the scanner focuses only on relevant code types.

### US003a - Manual Targeting Mode
**As a** warehouse user,  
**I want** to tap a code in a cluttered image,  
**so that** I ensure the correct item is decoded.

### US003b - Manual Targeting Mode
**As a** warehouse user,  
**I want** the app to highlight what it's "considering", and in case of shapes like the vertical container, helps me aim,  
**so that** I ensure I am aiming at the right area.

### US004 - Offline Operation
**As a** mobile user in remote areas and inside thick steel containers,  
**I want** the scanner to work entirely offline,  
**so that** I can use it in no-connectivity environments.

### US005 - Multi-Code Detection
**As a** quality control agent,  
**I want** the app to decode multiple codes in one frame,  
**so that** I can scan quickly and efficiently.

### US006 - Reusable Module
**As a** Cargosnap (internal!) developer of another React-Native Android/iOS app,  
**I want** to install the universal scanner plugin easily,  
**so that** I can integrate scanning without custom logic.

### US007 - Input Configuration
**As a** developer,  
**I want** to provide a structured input DTO,  
**so that** the scanner behaves based on configured scan modes, regex filters, and manual input flags.

### US008 - Verbose Debugging & Output
**As a** developer or tester,  
**I want** detailed logs and extended output data,  
**so that** I can trace processing steps, timings, and intermediate results during development.

---

## Features

- 🎯 **11 Supported Code Types**: QR codes, barcodes, license plates, container codes, and more
- 🚀 **Real-time Performance**: Native C++ core with hardware acceleration
- 📱 **Offline-First**: All ML inference runs on-device
- 🔌 **Plugin Architecture**: Easy integration with react-native-vision-camera
- 🎨 **Visual Feedback**: Bounding boxes and confidence scores
- 📋 **Domain-Specific Validation**: ISO standards compliance for containers, VINs, etc.

## Supported Code Types

| Type | Description | Example |
|------|-------------|---------|
| `code_qr_barcode` | Linear barcodes | EAN-13, Code128, UPC, 2D QR codes | Product links|
| `code_license_plate` | Vehicle plates | ABC-123, 12-AB-34 |
| `code_container_h` | ISO 6346 horizontal | MSCU1234567 |
| `code_container_v` | ISO 6346 vertical | MSCU1234567 |
| `text_printed` | General OCR text | Signs, labels |
| `code_seal` | Security seal codes | E12345678 |
| `code_lcd_display` | LCD/7-segment displays | Digital meters |
| `code_rail_wagon` | Railway car IDs | 33 87 4950 123-4 |
| `code_air_container` | Air cargo ULD codes | AKE12345AA |
| `code_vin` | Vehicle ID numbers | 1HGCM82633A004352 |

## Architecture

```
┌─────────────────────────────────────┐
│      React Native / TypeScript      │
│        (UI & Business Logic)        │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│    VisionCamera Frame Processor     │
│         (Plugin Interface)          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│        Native C++ Core              │
│  ┌─────────────┬─────────────────┐  │
│  │   YOLO v8   │     MLKit       │  │
│  │ (Detection) │  (Recognition)  │  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
```

## Installation

```bash
# Install the package
yarn add react-native-universal-scanner

# iOS specific
cd ios && pod install

# Configure metro.config.js for ONNX models
module.exports = {
  resolver: {
    assetExts: ['onnx', 'tflite', ...Platform.defaults.assetExts],
  },
};
```

## Basic Usage

```typescript
import { UniversalScanner, type ScanResult } from 'react-native-universal-scanner';
import { Camera, useCameraDevice } from 'react-native-vision-camera';

function ScannerScreen() {
  const device = useCameraDevice('back');
  
  const handleScan = (results: ScanResult[]) => {
    results.forEach(result => {
      console.log(`Found ${result.type}: ${result.value} (${result.confidence}%)`);
    });
  };

  if (!device) return <LoadingView />;

  return (
    <UniversalScanner
      camera={device}
      enabledTypes={['code_qr', 'code_container_h', 'text_printed']}
      onScan={handleScan}
      style={StyleSheet.absoluteFill}
    />
  );
}
```

## Advanced Configuration

```typescript
<UniversalScanner
  // Specify which code types to detect
  enabledTypes={['code_qr_barcode', 'code_license_plate','code_seal']}
  
  // Add regex validation per type
  regexPerType={{
    'code_container_h': [/^[A-Z]{4}\d{7}$/],
    'code_license_plate': [/^[A-Z]{2}-\d{2}-[A-Z]{2}$/]
  }}
  
  // Enable manual tap-to-scan mode
  manualMode={true}
  
  // Get detailed debug info
  verbose={true}
  
  // Callbacks
  onScan={handleScan}
  onError={handleError}
/>
```

## Frame Processor Usage

For custom processing pipelines:

```typescript
import { useScannerFrameProcessor } from 'react-native-universal-scanner';

const frameProcessor = useScannerFrameProcessor({
  enabledTypes: ['code_qr'],
  onScan: (results) => {
    'worklet';
    // Process results in the worklet context
    console.log(results);
  }
});

return <Camera frameProcessor={frameProcessor} {...props} />;
```

## Performance Optimization

- **GPU Acceleration**: Automatically uses Metal (iOS) and OpenGL ES (Android)
- **Adaptive Resolution**: Dynamically adjusts processing resolution based on device
- **Smart Caching**: Reuses detection results across frames when possible
- **Native Threading**: Heavy processing off the main thread

## Requirements

- React Native 0.71+
- react-native-vision-camera 3.x
- iOS 13+ / Android 6+
- For iOS: Xcode 14+ with Swift support
- For Android: Kotlin 1.6+, minSdkVersion 23

## Development

```bash
# Clone the repo
git clone https://github.com/yourusername/react-native-universal-scanner

# Install dependencies
yarn install

# Run the example app
cd example
yarn ios # or yarn android
```

## Acknowledgments

- Built on [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera)
- Uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- Leverages [MLKit](https://developers.google.com/ml-kit) for text recognition