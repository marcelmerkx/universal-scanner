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
  
  // Find EXACT 1280x720 format that our coordinate transformation expects
  const format = React.useMemo(() => {
    if (!device?.formats) return undefined
    
    // Look for EXACT 1280x720 format (for the time being; later more tolerant)
    const requiredFormat = device.formats.find(format => {
      return format.videoWidth === 1280 && format.videoHeight === 720
    })
    
    if (!requiredFormat) {
      console.error('CAMERA FORMAT ERROR: Could not find EXACT 1280x720 format!')
      console.error('Available formats:', device.formats.map(f => `${f.videoWidth}x${f.videoHeight} (${(f.videoWidth/f.videoHeight).toFixed(2)}:1)`))
      throw new Error('Required camera format not available. Need EXACT 1280x720 format for coordinate transformation.')
    }
    
    return requiredFormat
  }, [device?.formats])
  
  React.useEffect(() => {
    console.log('Screen dimensions:', screenDimensions)
  }, [screenDimensions])
  
  React.useEffect(() => {
    if (format) {
      console.log('Selected camera format:', {
        width: format.videoWidth,
        height: format.videoHeight,
        aspectRatio: (format.videoWidth / format.videoHeight).toFixed(3),
        fps: format.maxFps
      })
    }
  }, [format])
  
  // Load ONNX model for JavaScript processing (switching from native to test)
  // const onnxModel = useOnnxModel(require('../assets/unified-detection-v7.onnx'), 'cpu')
  // const { resize } = useResizePlugin()
  
  // Initialize VisionCamera plugin (keep for comparison)
  const universalScanner = VisionCameraProxy.initFrameProcessorPlugin('universalScanner')
  
  React.useEffect(() => {
    console.log('universalScanner plugin initialized:', typeof universalScanner, universalScanner)
  }, [universalScanner])
  
  // React.useEffect(() => {
  //   console.log('ONNX model state:', onnxModel.state)
  //   console.log('ONNX model object:', onnxModel.model)
  //   if (onnxModel.state === 'loaded') {
  //     console.log('ONNX model loaded successfully for JavaScript processing!')
  //     console.log('Model methods:', onnxModel.model ? Object.keys(onnxModel.model) : 'model is null')
  //   } else if (onnxModel.state === 'error') {
  //     console.error('ONNX model loading failed:', onnxModel.error)
  //   }
  // }, [onnxModel.state, onnxModel.model])
  
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
          
          console.log(`Raw C++ coordinates: (${onnxX},${onnxY}) size=${onnxW}×${onnxH}`)
          
          // ROTATION-AWARE COORDINATE TRANSFORMATION
          // Account for: Camera → 90°CW rotation → padding → ONNX → screen crop
          
          const modelSize = 640
          const cameraFormat = format || { videoWidth: 1280, videoHeight: 720 }
          
          // After 90° CW rotation: 1280×720 becomes 720×1280
          const rotatedCameraWidth = cameraFormat.videoHeight   // 720
          const rotatedCameraHeight = cameraFormat.videoWidth   // 1280
          
          console.log(`Camera: ${cameraFormat.videoWidth}×${cameraFormat.videoHeight} → Rotated: ${rotatedCameraWidth}×${rotatedCameraHeight}`)
          console.log(`Screen: ${screenDimensions.width}×${screenDimensions.height}`)
          console.log(`Model: ${modelSize}×${modelSize}`)
          
          // Step 1: Understand padding in ONNX space
          // 720×1280 image needs to fit in 640×640
          // Scale factor to fit: min(640/720, 640/1280) = 0.5
          const onnxScale = Math.min(modelSize / rotatedCameraWidth, modelSize / rotatedCameraHeight)
          const scaledWidth = rotatedCameraWidth * onnxScale    // 720 * 0.5 = 360
          const scaledHeight = rotatedCameraHeight * onnxScale  // 1280 * 0.5 = 640
          
          // Padding applied TOP-LEFT aligned (C++ WhitePadding implementation)
          const paddingLeft = 0     // Image starts at x=0
          const paddingTop = 0      // Image starts at y=0
          
          console.log(`ONNX padding: left=${paddingLeft}, top=${paddingTop}, scale=${onnxScale.toFixed(3)}`)
          
          // Step 2: Convert ONNX center coordinates to rotated camera coordinates
          // IMPORTANT: ONNX returns CENTER coordinates (x_center, y_center) in YOLO format
          // We need to convert these center coordinates properly
          
          // First, check if detection center is within the padded content area
          if (onnxX < paddingLeft || onnxX > (paddingLeft + scaledWidth)) {
            console.log(`Detection center X (${onnxX}) outside content area [${paddingLeft}, ${paddingLeft + scaledWidth}]`)
          }
          
          // Map from padded ONNX space to original rotated camera space
          // These are CENTER coordinates, so we map the center point
          const contentCenterX = onnxX - paddingLeft  // Center position within scaled content
          const contentCenterY = onnxY - paddingTop   // Center position within scaled content
          
          const unpaddedCenterX = contentCenterX / onnxScale  // Scale back to original size
          const unpaddedCenterY = contentCenterY / onnxScale
          const unpaddedW = onnxW / onnxScale
          const unpaddedH = onnxH / onnxScale
          
          // Convert from center coordinates to top-left for easier processing
          const unpaddedX = unpaddedCenterX - unpaddedW / 2
          const unpaddedY = unpaddedCenterY - unpaddedH / 2
          
          console.log(`Unpadded center: (${unpaddedCenterX.toFixed(1)},${unpaddedCenterY.toFixed(1)}) -> top-left: (${unpaddedX.toFixed(1)},${unpaddedY.toFixed(1)}) size=${unpaddedW.toFixed(1)}×${unpaddedH.toFixed(1)}`)
          
          // Step 3: Calculate screen scaling (resizeMode="contain") 
          // VisionCamera scales the rotated camera to fit within screen (no cropping)
          const screenAspect = screenDimensions.width / screenDimensions.height
          const rotatedCameraAspect = rotatedCameraWidth / rotatedCameraHeight
          
          let displayScale, displayOffsetX = 0, displayOffsetY = 0
          let displayWidth, displayHeight
          
          if (screenAspect > rotatedCameraAspect) {
            // Screen is wider - camera fits height, add padding on sides
            displayScale = screenDimensions.height / rotatedCameraHeight
            displayWidth = rotatedCameraWidth * displayScale
            displayHeight = screenDimensions.height
            displayOffsetX = (screenDimensions.width - displayWidth) / 2
            displayOffsetY = 0
          } else {
            // Screen is taller - camera fits width, add padding top/bottom  
            displayScale = screenDimensions.width / rotatedCameraWidth
            displayWidth = screenDimensions.width
            displayHeight = rotatedCameraHeight * displayScale
            displayOffsetX = 0
            displayOffsetY = (screenDimensions.height - displayHeight) / 2
          }
          
          console.log(`Contain mode: scale=${displayScale.toFixed(3)}, display=${displayWidth.toFixed(1)}x${displayHeight.toFixed(1)}`)
          console.log(`Display offsets: X=${displayOffsetX.toFixed(1)}, Y=${displayOffsetY.toFixed(1)}`)
          
          // Step 4: Map to screen coordinates (no cropping, just scaling + offset)
          // Use CENTER coordinates for proper mapping to screen space
          const normalizedX = unpaddedCenterX / rotatedCameraWidth
          const normalizedY = unpaddedCenterY / rotatedCameraHeight
          
          // Direct mapping - no cropping involved
          const screenCenterX = normalizedX * displayWidth + displayOffsetX
          const screenCenterY = normalizedY * displayHeight + displayOffsetY
          const screenW = (unpaddedW / rotatedCameraWidth) * displayWidth
          const screenH = (unpaddedH / rotatedCameraHeight) * displayHeight
          
          // Convert from center coordinates to top-left for React Native View
          const boxLeft = screenCenterX - screenW / 2
          const boxTop = screenCenterY - screenH / 2
          
          console.log(`=== CONTAIN MODE MAPPING ===`)
          console.log(`Normalized in camera: (${normalizedX.toFixed(3)},${normalizedY.toFixed(3)})`)
          console.log(`Screen center: (${screenCenterX.toFixed(1)},${screenCenterY.toFixed(1)})`)
          console.log(`Final box: left=${boxLeft.toFixed(1)}, top=${boxTop.toFixed(1)}, size=${screenW.toFixed(1)}×${screenH.toFixed(1)}`)
          
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
            format={format}
            resizeMode="contain"
          />
          <View style={styles.overlay}>
            <Text style={styles.fps}>FPS: {fps}</Text>
            <Text style={styles.title}>Native C++ ONNX Detection</Text>
            <Text style={styles.subtitle}>
              Real-time license plate detection with bounding boxes
            </Text>
            {renderResult()}
          </View>
          
          {/* Debug Grid Overlay
          <View style={styles.debugGrid} pointerEvents="none">
            {Array.from({ length: 11 }, (_, i) => (
              <View key={`h-${i}`} style={[styles.gridLineHorizontal, { top: `${i * 10}%` }]}>
                <Text style={styles.gridLabel}>{i * 10}%</Text>
              </View>
            ))}
            {Array.from({ length: 11 }, (_, i) => (
              <View key={`v-${i}`} style={[styles.gridLineVertical, { left: `${i * 10}%` }]}>
                <Text style={styles.gridLabelVertical}>{i * 10}%</Text>
              </View>
            ))}
          </View> */}

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
    backgroundColor: 'rgba(41, 230, 17, 0.09)',
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
  debugGrid: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
  },
  gridLineHorizontal: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  gridLineVertical: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  gridLabel: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 8,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    paddingHorizontal: 2,
    paddingVertical: 1,
  },
  gridLabelVertical: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 8,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    paddingHorizontal: 2,
    paddingVertical: 1,
    transform: [{ rotate: '90deg' }],
  },
})