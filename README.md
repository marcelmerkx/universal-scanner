# Universal Scanner

A high-performance React Native plugin for real-time detection and decoding of multiple code types using computer vision and machine learning.

## Overview

Universal Scanner is a `react-native-vision-camera` plugin that combines YOLO object detection with MLKit text recognition to scan and decode various types of codes and text in real-time. Built with a hybrid native architecture for optimal performance.

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
| `code_qr` | 2D QR codes | Product links, WiFi configs |
| `code_barcode_1d` | Linear barcodes | EAN-13, Code128, UPC |
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
  enabledTypes={['code_qr', 'code_barcode_1d', 'code_license_plate']}
  
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Built on [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera)
- Uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- Leverages [MLKit](https://developers.google.com/ml-kit) for text recognition