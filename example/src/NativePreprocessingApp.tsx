import * as React from 'react'
import { StyleSheet, View, Text, ActivityIndicator, ScrollView, Dimensions, TouchableOpacity } from 'react-native'
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
import { CODE_DETECTION_TYPES } from './CodeDetectionTypes'
import { renderSimplifiedBoundingBoxes } from './SimplifiedBoundingBoxes'
// import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated'

// Type definitions for scanner results
interface Detection {
  type: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
  model: string
}

interface DetectionResult {
  detections?: Detection[]
  error?: string
}

interface AnimatedBoundingBoxProps {
  detection: Detection
  boxLeft: number
  boxTop: number
  screenW: number
  screenH: number
  previousPosition?: { x: number; y: number; width: number; height: number }
}

const AnimatedBoundingBox: React.FC<AnimatedBoundingBoxProps> = ({ 
  detection, 
  boxLeft, 
  boxTop, 
  screenW, 
  screenH, 
  previousPosition 
}) => {
  // Simplified non-animated version for now
  return (
    <View style={[
      styles.boundingBox,
      {
        left: Math.max(0, boxLeft),
        top: Math.max(0, boxTop),
        width: Math.max(20, screenW),
        height: Math.max(20, screenH),
      }
    ]}>
      <Text style={styles.boundingBoxLabel}>
        {detection.type.replace('code_', '')}: {(Number(detection.confidence) * 100).toFixed(0)}%
      </Text>
    </View>
  )
}

export default function NativePreprocessingApp(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const [lastResult, setLastResult] = React.useState<any>(null)
  const [fps, setFps] = React.useState(0)
  const [debugImages, setDebugImages] = React.useState(false)
  const [useTflite, setUseTflite] = React.useState(false) // Toggle for A/B testing - DEFAULT TO ONNX (TFLite crashes in C++)
  const screenDimensions = Dimensions.get('window')
  const [previousDetections, setPreviousDetections] = React.useState<Map<string, any>>(new Map())
  
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
      
      // TO NATIVE C++ MODE FOR TENSOR DUMP
      try {
        if (!universalScanner || typeof universalScanner.call !== 'function') {
          return
        }
        
        // Run the universal scanner with native preprocessing using VisionCamera plugin
        const result = universalScanner.call(frame, {
          enabledTypes: [
            CODE_DETECTION_TYPES.QR_BARCODE, 
            CODE_DETECTION_TYPES.CONTAINER_H, 
            CODE_DETECTION_TYPES.CONTAINER_V, 
            CODE_DETECTION_TYPES.LICENSE_PLATE,
            CODE_DETECTION_TYPES.SEAL
          ],
          debugImages: debugImages,
          useTflite: useTflite, // Enable TFLite backend for A/B testing
        }) as DetectionResult
        
        if (result?.error) {
          console.error('Scanner error:', result.error)
        } else if (result?.detections && result.detections.length > 0) {
          onScanResult(result)
        } else {
          // No detections found - this is normal, don't call callback
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
  
  // Update previous detection positions for smooth transitions
  React.useEffect(() => {
    if (lastResult?.detections) {
      const newPreviousDetections = new Map()
      lastResult.detections.forEach((detection: any, index: number) => {
        // Use the same key generation logic as in render
        const detectionKey = `${detection.type}-${Math.round(detection.x)}-${Math.round(detection.y)}`
        // Store current position as previous for next render
        newPreviousDetections.set(detectionKey, {
          x: detection.x,
          y: detection.y,
          width: detection.width,
          height: detection.height
        })
      })
      setPreviousDetections(newPreviousDetections)
    }
  }, [lastResult])

  const renderBoundingBoxes = () => {
    return renderSimplifiedBoundingBoxes(
      lastResult,
      format,
      screenDimensions,
      previousDetections,
      AnimatedBoundingBox,
      styles
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
          />
          <View style={styles.overlay}>
            <Text style={styles.fps}>FPS: {fps}</Text>
            <Text style={styles.title}>Native C++ {useTflite ? 'TFLite' : 'ONNX'} Detection</Text>
            <Text style={styles.subtitle}>
              Real-time object detection with bounding boxes
            </Text>
            <View style={styles.buttonContainer}>
              <TouchableOpacity 
                style={styles.debugButton} 
                onPress={() => setDebugImages(!debugImages)}
              >
                <Text style={styles.debugButtonText}>
                  Debug Images: {debugImages ? 'ON' : 'OFF'}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.debugButton, styles.tfliteButton]} 
                onPress={() => setUseTflite(!useTflite)}
              >
                <Text style={styles.debugButtonText}>
                  Backend: {useTflite ? 'TFLite' : 'ONNX'}
                </Text>
              </TouchableOpacity>
            </View>
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
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 10,
  },
  debugButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  tfliteButton: {
    backgroundColor: 'rgba(255, 128, 0, 0.3)',
  },
  debugButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
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
    backgroundColor: 'rgba(0, 0, 0, 0.1)', // Much more transparent to see camera alignment
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