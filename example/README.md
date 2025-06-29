# Universal Scanner Example App

This is the demo application for the `react-native-universal-scanner` plugin, showcasing real-time code detection and scanning capabilities.

## Overview

The example app demonstrates:
- Real-time camera feed with frame processing
- Multi-code type detection (QR, barcodes, text, containers, etc.)
- Visual bounding boxes with confidence scores
- Performance metrics and debugging tools

## Setup

### Prerequisites

- Node.js 16+
- Yarn or npm
- For iOS: macOS with Xcode 14+
- For Android: Android Studio with SDK 23+

### Installation

```bash
# From the example directory
yarn install

# iOS only
cd ios && pod install && cd ..
```

### Running the App

```bash
# Android
yarn android

# iOS (requires macOS)
yarn ios
```

## Features Demonstrated

### 1. Basic Scanner
Simple QR and barcode scanning with automatic detection.

### 2. Container Scanner
Specialized mode for ISO 6346 container code detection with validation.

### 3. Multi-Type Scanner
Simultaneous detection of multiple code types with filtering options.

### 4. Manual Mode
Tap-to-scan functionality for crowded scenes.

### 5. Debug Mode
Verbose output showing:
- Frame processing times
- Detection confidence scores
- Bounding box coordinates
- Model inference details

## Project Structure

```
example/
├── src/
│   ├── App.tsx           # Main app component
│   ├── screens/          # Different scanner demos
│   ├── components/       # Reusable UI components
│   └── utils/           # Helper functions
├── android/             # Android native code
├── ios/                # iOS native code
└── assets/             # Test images and models
```

## Configuration

The app includes several configuration options:

```typescript
// In App.tsx
const SCANNER_CONFIG = {
  enabledTypes: ['code_qr', 'code_qr_barcode', 'code_container_h'],
  verbose: true,
  manualMode: false,
  regexPerType: {
    'code_container_h': [/^[A-Z]{4}\d{7}$/]
  }
};
```

## Testing Different Scenarios

### Good Lighting
Test with well-lit, clear codes to verify basic functionality.

### Low Light
Enable device torch to test low-light performance.

### Multiple Codes
Test with multiple codes in frame to verify detection filtering.

### Motion Blur
Move camera while scanning to test motion compensation.

### Different Angles
Test detection at various angles (0°, 45°, 90°).

## Performance Metrics

The app displays real-time performance metrics:
- FPS (Frames Per Second)
- Detection latency (ms)
- Memory usage
- CPU utilization

## Troubleshooting

### Build Issues

**Android**: Clean and rebuild
```bash
cd android && ./gradlew clean && cd ..
yarn android
```

**iOS**: Clean build folder
```bash
cd ios && xcodebuild clean && pod install && cd ..
yarn ios
```

### Camera Permissions

Ensure camera permissions are granted in device settings.

### Performance Issues

- Reduce camera resolution
- Enable frame skipping
- Disable verbose mode

## Development

### Adding New Scanner Modes

1. Create new screen in `src/screens/`
2. Configure scanner with specific `enabledTypes`
3. Add navigation entry in `App.tsx`

### Custom Frame Processing

```typescript
const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  const results = scanFrame(frame);
  // Custom processing logic
}, []);
```

## Known Limitations

- iOS Simulator: Camera not available, use real device
- Android Emulator: Limited performance, real device recommended
- Some ONNX operations may not be supported on all devices

## Contributing

See main project [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.