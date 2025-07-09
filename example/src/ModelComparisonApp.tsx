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
import { ScannerResult, useOnnxModel, useTensorflowModel } from 'react-native-fast-tflite'
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
  value?: string
}

interface DetectionResult {
  detections?: Detection[]
  ocr_results?: Array<{
    type: string
    value: string
    confidence: number
    model: string
  }>
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
        {detection.value ? ` - ${detection.value}` : ''}
      </Text>
    </View>
  )
}

export default function ModelComparisonApp(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const [lastResult, setLastResult] = React.useState<any>(null)
  const [fps, setFps] = React.useState(0)
  const [debugImages, setDebugImages] = React.useState(true) // Default to true
  const screenDimensions = Dimensions.get('window')
  const [previousDetections, setPreviousDetections] = React.useState<Map<string, any>>(new Map())
  const [modelSize, setModelSize] = React.useState<320 | 416 | 640>(320)
  const [inferenceTime, setInferenceTime] = React.useState(0)
  
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
  
  // Load models for JavaScript processing - TFLite removed, ONNX runs natively
  // const tfliteModel = useTensorflowModel(require('../assets/unified-detection-v7_float32.tflite'))
  // const onnxModel = useOnnxModel(require('../assets/unified-detection-v7.onnx'), 'cpu')
  const { resize } = useResizePlugin()
  
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
  
  // Process TFLite YOLO output
  const processTfliteOutput = (output: Float32Array): Detection[] => {
    'worklet'
    
    // Non-Maximum Suppression (inline for worklet)
    const applyNMS = (detections: Detection[], iouThreshold: number): Detection[] => {
      if (detections.length === 0) return []
      
      // Sort by confidence
      const sorted = [...detections].sort((a, b) => b.confidence - a.confidence)
      const selected: Detection[] = []
      
      for (const current of sorted) {
        let shouldSelect = true
        
        for (const existing of selected) {
          // Calculate IoU (Intersection over Union)
          const x1 = Math.max(current.x, existing.x)
          const y1 = Math.max(current.y, existing.y)
          const x2 = Math.min(current.x + current.width, existing.x + existing.width)
          const y2 = Math.min(current.y + current.height, existing.y + existing.height)
          
          const intersectionWidth = Math.max(0, x2 - x1)
          const intersectionHeight = Math.max(0, y2 - y1)
          const intersectionArea = intersectionWidth * intersectionHeight
          
          const currentArea = current.width * current.height
          const existingArea = existing.width * existing.height
          const unionArea = currentArea + existingArea - intersectionArea
          
          const iou = intersectionArea / unionArea
          
          if (iou > iouThreshold && current.type === existing.type) {
            shouldSelect = false
            break
          }
        }
        
        if (shouldSelect) {
          selected.push(current)
        }
      }
      
      return selected
    }
    const detections: Detection[] = []
    const confidenceThreshold = 0.25
    const classNames = ['qr_barcode', 'container_h', 'container_v', 'license_plate', 'seal']
    
    const numPredictions = 3549
    
    for (let i = 0; i < numPredictions; i++) {
      const x = output[0 * numPredictions + i]
      const y = output[1 * numPredictions + i]
      const w = output[2 * numPredictions + i]
      const h = output[3 * numPredictions + i]
      
      let bestClass = -1
      let bestConf = 0
      
      for (let c = 0; c < 5; c++) {
        const score = output[(4 + c) * numPredictions + i]
        if (score > bestConf) {
          bestClass = c
          bestConf = score
        }
      }
      
      if (bestConf > confidenceThreshold) {
        // TFLite outputs normalized center coordinates
        // Convert to pixel coordinates in 416x416 space
        const centerX = x * 416
        const centerY = y * 416
        const boxWidth = w * 416
        const boxHeight = h * 416
        
        // Convert center to top-left coordinates
        const topLeftX = centerX - boxWidth / 2
        const topLeftY = centerY - boxHeight / 2
        
        // Map from 416x416 to 640x640 coordinate system
        // The model was trained on 640x640 images with 360x640 content area
        // But TFLite export uses 416x416, so we need to scale appropriately
        const scale = 640.0 / 416.0
        
        detections.push({
          type: 'code_' + classNames[bestClass],
          confidence: bestConf,
          x: topLeftX * scale,
          y: topLeftY * scale,
          width: boxWidth * scale,
          height: boxHeight * scale,
          model: 'unified-detection-v7.tflite'
        })
      }
    }
    
    // Apply Non-Maximum Suppression to reduce duplicates
    return applyNMS(detections, 0.5)
  }

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
  
  const onScanResult = Worklets.createRunOnJS((result: any, inferenceMs: number) => {
    setLastResult(result)
    setInferenceTime(inferenceMs)
    console.log('Scan result:', result)
  })

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet'
      
      // Update FPS counter
      updateFps()
      
      try {
        const startTime = Date.now()
        
        // Native ONNX processing with model size
        if (!universalScanner || typeof universalScanner.call !== 'function') {
          return
        }
        
        const result = universalScanner.call(frame, {
          enabledTypes: [
            // CODE_DETECTION_TYPES.QR_BARCODE, 
            CODE_DETECTION_TYPES.CONTAINER_H, 
            CODE_DETECTION_TYPES.CONTAINER_V, 
            // CODE_DETECTION_TYPES.LICENSE_PLATE,
            // CODE_DETECTION_TYPES.SEAL
          ],
          debugImages: debugImages,
          modelSize: modelSize, // Pass model size to native
        }) as DetectionResult
        
        if (result?.error) {
          console.error('Scanner error:', result.error)
        } else if (result?.detections && result.detections.length > 0) {
          // Merge OCR results with detections
          if (result.ocr_results && result.ocr_results.length > 0) {
            console.log('OCR Results:', result.ocr_results)
            // For now, just add the first OCR result to the first detection
            // In a real app, you'd match by type or position
            result.detections.forEach((detection, idx) => {
              const ocrResult = result.ocr_results?.find(ocr => ocr.type === detection.type)
              if (ocrResult) {
                detection.value = ocrResult.value
                console.log(`Merged OCR value "${ocrResult.value}" into detection of type ${detection.type}`)
              }
            })
          }
          const inferenceMs = Date.now() - startTime
          onScanResult(result, inferenceMs)
        }
      } catch (error) {
        console.log('VisionCamera universalScanner error:', error)
        console.log('Error type:', typeof error)
        console.log('Error message:', error?.message)
        console.log('Error stack:', error?.stack)
      }
    },
    [universalScanner, debugImages, modelSize]
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
    if (!lastResult?.detections) return null
    
    // Handle different model sizes (320, 416, 640)
    const modelSizeUsed = modelSize // Current model size
    
    // Model space dimensions vary but content area maintains aspect ratio
    // For 640: content is 360x640 (after rotation)
    // For 416: content is 234x416 (scaled proportionally)  
    // For 320: content is 180x320 (scaled proportionally)
    const contentWidth = modelSizeUsed * (360.0 / 640.0)
    const contentHeight = modelSizeUsed
    
    return (
      <View style={styles.boundingBoxContainer}>
        {lastResult.detections.map((detection: any, index: number) => {
          const onnxX = parseInt(detection.x)
          const onnxY = parseInt(detection.y)
          const onnxW = parseInt(detection.width)
          const onnxH = parseInt(detection.height)
          
          // Map from model space to screen space
          const normalizedX = onnxX / contentWidth
          const normalizedY = onnxY / contentHeight
          
          const screenCenterX = normalizedX * screenDimensions.width
          const screenCenterY = normalizedY * screenDimensions.height
          const screenW = (onnxW / contentWidth) * screenDimensions.width
          const screenH = (onnxH / contentHeight) * screenDimensions.height
          
          // Convert center to top-left
          const boxLeft = screenCenterX - screenW / 2
          const boxTop = screenCenterY - screenH / 2
          
          // Use the same key generation logic as original
          const detectionKey = `${detection.type}-${Math.round(detection.x)}-${Math.round(detection.y)}`
          const previousPosition = previousDetections.get(detectionKey)
          
          return (
            <AnimatedBoundingBox
              key={detectionKey}
              detection={detection}
              boxLeft={boxLeft}
              boxTop={boxTop}
              screenW={screenW}
              screenH={screenH}
              previousPosition={previousPosition}
            />
          )
        })}
      </View>
    )
  }
  
  

  const renderResult = () => {
    if (!lastResult) return null
    
    return (
      <View style={styles.resultContainer}>
        <Text style={styles.resultTitle}>
          Detections: {lastResult.detections?.length || 0}
          {lastResult.detections && lastResult.detections[0]?.value && (
            <Text style={styles.ocrInline}> • OCR: {lastResult.detections[0].value}</Text>
          )}
        </Text>
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
            <Text style={styles.fps}>FPS: {fps} | {inferenceTime}ms</Text>
            <Text style={styles.title}>
              Native ONNX ({modelSize}x{modelSize}) • Debug: ON
            </Text>
            <View style={styles.buttonContainer}>
              {/* Debug button commented out - causes crashes when toggled
              <TouchableOpacity 
                style={styles.debugButton} 
                onPress={() => setDebugImages(!debugImages)}
              >
                <Text style={styles.debugButtonText}>
                  Debug Images: {debugImages ? 'ON' : 'OFF'}
                </Text>
              </TouchableOpacity> */}
              {/* <TouchableOpacity 
                style={[styles.debugButton, modelSize === 320 && styles.activeButton]} 
                onPress={() => setModelSize(320)}
              >
                <Text style={styles.debugButtonText}>
                  320x320
                </Text>
              </TouchableOpacity> */}
              {/* <TouchableOpacity 
                style={[styles.debugButton, modelSize === 416 && styles.activeButton]} 
                onPress={() => setModelSize(416)}
              >
                <Text style={styles.debugButtonText}>
                  416x416
                </Text>
              </TouchableOpacity> */}
              {/* <TouchableOpacity 
                style={[styles.debugButton, modelSize === 640 && styles.activeButton]} 
                onPress={() => setModelSize(640)}
              >
                <Text style={styles.debugButtonText}>
                  640x640
                </Text>
              </TouchableOpacity> */}
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
  ocrValue: {
    fontSize: 12,
    color: '#00FF00',
    fontWeight: 'bold',
    marginTop: 3,
  },
  ocrInline: {
    color: '#00FF00',
    fontWeight: 'bold',
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
  activeButton: {
    backgroundColor: 'rgba(0, 122, 255, 0.5)',
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