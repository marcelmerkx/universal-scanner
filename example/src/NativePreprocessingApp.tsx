import * as React from 'react'
import { StyleSheet, View, Text, ActivityIndicator, ScrollView, Dimensions } from 'react-native'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  Frame,
  VisionCameraProxy,
} from 'react-native-vision-camera'
import { ScannerResult, useOnnxModel } from 'react-native-fast-tflite'
import { useResizePlugin } from 'vision-camera-resize-plugin'
import { Worklets } from 'react-native-worklets-core'

export default function NativePreprocessingApp(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const [lastResult, setLastResult] = React.useState<any>(null)
  const [fps, setFps] = React.useState(0)
  const screenDimensions = Dimensions.get('window')
  
  React.useEffect(() => {
    console.log('Screen dimensions:', screenDimensions)
  }, [screenDimensions])
  
  // Load ONNX model for JavaScript processing (switching from native to test)
  const onnxModel = useOnnxModel(require('../assets/unified-detection-v7.onnx'), 'cpu')
  const { resize } = useResizePlugin()
  
  // Initialize VisionCamera plugin (keep for comparison)
  const universalScanner = VisionCameraProxy.initFrameProcessorPlugin('universalScanner')
  
  React.useEffect(() => {
    console.log('universalScanner plugin initialized:', typeof universalScanner, universalScanner)
  }, [universalScanner])
  
  React.useEffect(() => {
    console.log('ONNX model state:', onnxModel.state)
    console.log('ONNX model object:', onnxModel.model)
    if (onnxModel.state === 'loaded') {
      console.log('ONNX model loaded successfully for JavaScript processing!')
      console.log('Model methods:', onnxModel.model ? Object.keys(onnxModel.model) : 'model is null')
    } else if (onnxModel.state === 'error') {
      console.error('ONNX model loading failed:', onnxModel.error)
    }
  }, [onnxModel.state, onnxModel.model])
  
  // Track FPS
  const frameCount = React.useRef(0)
  const lastFpsUpdate = React.useRef(Date.now())
  
  const updateFps = Worklets.createRunOnJS(() => {
    frameCount.current++
    const now = Date.now()
    const elapsed = now - lastFpsUpdate.current
    
    if (elapsed >= 1000) {
      setFps(Math.round((frameCount.current * 1000) / elapsed))
      frameCount.current = 0
      lastFpsUpdate.current = now
    }
  })
  
  const onScanResult = Worklets.createRunOnJS((result: any) => {
    setLastResult(result)
    console.log('Scan result:', result)
  })

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet'
      
      // Update FPS counter
      updateFps()
      
      // BACK TO NATIVE C++ MODE FOR TENSOR DUMP
      try {
        if (!universalScanner || typeof universalScanner.call !== 'function') {
          return
        }
        
        // Run the universal scanner with native preprocessing using VisionCamera plugin
        const result = universalScanner.call(frame, {
          enabledTypes: ['code_qr_barcode', 'code_container_h', 'code_container_v', 'code_license_plate'],
          verbose: true,
        })
        
        if (result) {
          onScanResult(result)
        }
      } catch (error) {
        console.log('VisionCamera universalScanner error:', error)
        console.log('Error type:', typeof error)
        console.log('Error message:', error?.message)
        console.log('Error stack:', error?.stack)
      }
    },
    [universalScanner]
  )

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  const renderBoundingBoxes = () => {
    if (!lastResult?.detections) return null
    
    return (
      <View style={styles.boundingBoxContainer}>
        {lastResult.detections.map((detection: any, index: number) => {
          // C++ returns pixel coordinates in 640x640 ONNX space (center point + dimensions)
          const onnxX = parseInt(detection.x)      // Center X in 640x640 space
          const onnxY = parseInt(detection.y)      // Center Y in 640x640 space  
          const onnxW = parseInt(detection.width)  // Width in 640x640 space
          const onnxH = parseInt(detection.height) // Height in 640x640 space
          
          // The ONNX coordinates are inverted on both axes
          // Use UNIFORM scaling to maintain aspect ratio
          // The ONNX space is 640x640 square, but screen is not square
          const uniformScale = Math.min(screenDimensions.width / 640, screenDimensions.height / 640)
          
          // Calculate offsets to center the 640x640 space on screen
          const offsetX = (screenDimensions.width - 640 * uniformScale) / 2
          const offsetY = (screenDimensions.height - 640 * uniformScale) / 2
          
          // Flip and scale coordinates uniformly
          const screenX = (640 - onnxX) * uniformScale + offsetX
          const screenY = (640 - onnxY) * uniformScale + offsetY
          
          // Scale dimensions uniformly
          const screenW = onnxW * uniformScale
          const screenH = onnxH * uniformScale
          
          // Convert from center coordinates to top-left for React Native positioning
          const boxLeft = screenX - screenW / 2
          const boxTop = screenY - screenH / 2
          
          console.log(`=== COORDINATE TRANSFORMATION (UNIFORM SCALING) ===`)
          console.log(`ONNX coordinates (640x640): center=(${onnxX},${onnxY}), size=${onnxW}×${onnxH}`)
          console.log(`Screen dimensions: ${screenDimensions.width}×${screenDimensions.height}`)
          console.log(`Uniform scale: ${uniformScale.toFixed(3)} (maintains aspect ratio)`)
          console.log(`Offsets: X=${offsetX.toFixed(1)}, Y=${offsetY.toFixed(1)}`)
          console.log(`Flipped & scaled center: (${screenX.toFixed(1)},${screenY.toFixed(1)})`)
          console.log(`Scaled size: ${screenW.toFixed(1)}×${screenH.toFixed(1)}`)
          console.log(`Bounding box: left=${boxLeft.toFixed(1)}, top=${boxTop.toFixed(1)}`)
          
          return (
            <React.Fragment key={index}>
              {/* Main bounding box (green) */}
              <View
                style={[
                  styles.boundingBox,
                  {
                    left: Math.max(0, boxLeft),
                    top: Math.max(0, boxTop),
                    width: Math.max(20, screenW),
                    height: Math.max(20, screenH),
                  }
                ]}
              >
                <Text style={styles.boundingBoxLabel}>
                  {detection.type.replace('code_', '')}: {(parseFloat(detection.confidence) * 100).toFixed(0)}%
                </Text>
              </View>
            </React.Fragment>
          )
        })}
      </View>
    )
  }

  const renderResult = () => {
    if (!lastResult) return null
    
    return (
      <View style={styles.resultContainer}>
        <Text style={styles.resultTitle}>Detections: {lastResult.detections?.length || 0}</Text>
        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          {lastResult.detections && lastResult.detections.map((detection: any, index: number) => (
            <View key={index} style={styles.detection}>
              <Text style={styles.detectionType}>{detection.type.replace('code_', '')}</Text>
              <Text style={styles.detectionValue}>
                {(parseFloat(detection.confidence) * 100).toFixed(0)}% • {parseInt(detection.width)}×{parseInt(detection.height)}
              </Text>
              <Text style={styles.detectionValue}>
                pos: {parseInt(detection.x)},{parseInt(detection.y)} • {detection.model}
              </Text>
            </View>
          ))}
        </ScrollView>
      </View>
    )
  }

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <>
          <Camera
            device={device}
            style={StyleSheet.absoluteFill}
            isActive={true}
            frameProcessor={frameProcessor}
            pixelFormat="yuv"
            enableFpsGraph={true}
          />
          <View style={styles.overlay}>
            <Text style={styles.fps}>FPS: {fps}</Text>
            <Text style={styles.title}>Native C++ ONNX Detection</Text>
            <Text style={styles.subtitle}>
              Real-time license plate detection with bounding boxes
            </Text>
            {renderResult()}
          </View>
          
          {/* Bounding Box Overlay */}
          {renderBoundingBoxes()}
          
          {/* Viewfinder Overlay */}
          <View style={styles.viewfinderOverlay}>
            {/* Top overlay */}
            <View style={[styles.overlaySection, { height: '5%' }]} />
            
            {/* Middle section with side dimming */}
            <View style={[styles.middleSection, { height: '80%' }]}>
              {/* Left side dim */}
              <View style={[styles.overlaySection, { width: '5%', height: '100%' }]} />
              
              {/* Clear viewfinder area */}
              <View style={styles.viewfinderContainer}>
                <View style={styles.viewfinderFrame} />
              </View>
              
              {/* Right side dim */}
              <View style={[styles.overlaySection, { width: '5%', height: '100%' }]} />
            </View>
            
            {/* Bottom overlay */}
            <View style={[styles.overlaySection, { height: '20%' }]} />
          </View>
        </>
      ) : (
        <View style={styles.centerContainer}>
          <Text style={styles.title}>Camera permission required</Text>
        </View>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginTop: 50,
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: -1, height: 1 },
    textShadowRadius: 10,
  },
  subtitle: {
    fontSize: 16,
    color: 'white',
    textAlign: 'center',
    marginTop: 10,
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: -1, height: 1 },
    textShadowRadius: 10,
  },
  fps: {
    position: 'absolute',
    top: 50,
    right: 20,
    fontSize: 18,
    fontWeight: 'bold',
    color: 'yellow',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: -1, height: 1 },
    textShadowRadius: 10,
  },
  resultContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    padding: 10,
    paddingBottom: 20,
    maxHeight: 120,
  },
  scrollView: {
    flex: 1,
  },
  resultTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  detection: {
    marginBottom: 5,
    paddingBottom: 5,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.2)',
  },
  detectionType: {
    fontSize: 12,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  detectionValue: {
    fontSize: 12,
    color: 'white',
    marginTop: 1,
  },
  detectionConfidence: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  paddingInfo: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  viewfinderOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'column',
    pointerEvents: 'none',
  },
  overlaySection: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  middleSection: {
    flexDirection: 'row',
    width: '100%',
  },
  viewfinderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  viewfinderFrame: {
    width: '100%',
    height: '100%',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.8)',
    borderRadius: 8,
    backgroundColor: 'transparent',
  },
  boundingBoxContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 3,
    borderColor: '#00FF00',
    backgroundColor: 'transparent',
    borderRadius: 4,
  },
  boundingBoxLabel: {
    position: 'absolute',
    top: -25,
    left: 0,
    backgroundColor: 'rgba(0, 255, 0, 0.8)',
    color: 'black',
    fontSize: 12,
    fontWeight: 'bold',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 3,
    overflow: 'hidden',
  },
})