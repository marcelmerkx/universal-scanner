import * as React from 'react'
import { 
  StyleSheet, 
  View, 
  Text, 
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  ScrollView
} from 'react-native'
import {
  TensorflowModel,
  useTensorflowModel,
  OnnxModel,
  useOnnxModel,
  universalScanner,
} from 'react-native-fast-tflite'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera'
import { useResizePlugin } from 'vision-camera-resize-plugin'
import { Worklets } from 'react-native-worklets-core'

type InferenceMode = 'onnx-640' | 'onnx-416' | 'onnx-320' | 'tflite-js' | 'native-onnx' | 'native-tflite'

interface Detection {
  className: string
  confidence: number
  bbox: { x: number; y: number; w: number; h: number }
}

const { width: screenW, height: screenH } = Dimensions.get('window')

export default function EnhancedSwitchableApp(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const [mode, setMode] = React.useState<InferenceMode>('native-onnx')
  const [detections, setDetections] = React.useState<Detection[]>([])
  const [fps, setFps] = React.useState(0)
  const [inferenceTime, setInferenceTime] = React.useState(0)
  
  // Load models
  const tfliteModel = useTensorflowModel(require('../assets/unified-detection-v7_float32.tflite'))
  const onnxModel = useOnnxModel(require('../assets/unified-detection-v7.onnx'), 'cpu')
  
  // Resize plugin
  const { resize } = useResizePlugin()
  
  // FPS tracking
  const frameCount = React.useRef(0)
  const lastFpsUpdate = React.useRef(Date.now())
  const inferenceStart = React.useRef(0)
  
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
  
  const updateDetections = Worklets.createRunOnJS((dets: Detection[], time: number) => {
    setDetections(dets)
    setInferenceTime(time)
  })
  
  // Process YOLO output
  const processYoloOutput = (output: Float32Array | any, modelSize: number): Detection[] => {
    'worklet'
    const detections: Detection[] = []
    const confidenceThreshold = 0.25
    const classNames = ['qr_barcode', 'container_h', 'container_v', 'license_plate', 'seal']
    
    // Handle both Float32Array and nested array formats
    let outputArray: Float32Array
    if (output instanceof Float32Array) {
      outputArray = output
    } else if (output.data && output.data instanceof Float32Array) {
      outputArray = output.data
    } else if (output.data) {
      // ONNX output with nested arrays
      try {
        const data = output.data as number[][][]
        const preds2d = data[0] // Shape: [9, 3549]
        // Flatten the 2D array
        const flattened: number[] = []
        for (let i = 0; i < preds2d.length; i++) {
          for (let j = 0; j < preds2d[i].length; j++) {
            flattened.push(preds2d[i][j])
          }
        }
        outputArray = new Float32Array(flattened)
      } catch (e) {
        console.log('Failed to process ONNX output:', e)
        return []
      }
    } else {
      console.log('Unknown output format')
      return []
    }
    
    const numPredictions = 3549
    
    for (let i = 0; i < numPredictions; i++) {
      const x = outputArray[0 * numPredictions + i]
      const y = outputArray[1 * numPredictions + i]
      const w = outputArray[2 * numPredictions + i]
      const h = outputArray[3 * numPredictions + i]
      
      let bestClass = -1
      let bestConf = 0
      
      for (let c = 0; c < 5; c++) {
        const score = outputArray[(4 + c) * numPredictions + i]
        if (score > bestConf) {
          bestClass = c
          bestConf = score
        }
      }
      
      if (bestConf > confidenceThreshold) {
        detections.push({
          className: classNames[bestClass],
          confidence: bestConf,
          bbox: {
            x: (x - w/2) * modelSize,
            y: (y - h/2) * modelSize,
            w: w * modelSize,
            h: h * modelSize
          }
        })
      }
    }
    
    return detections
  }
  
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      
      updateFps()
      
      try {
        inferenceStart.current = Date.now()
        let detectionResults: Detection[] = []
        
        if (mode === 'tflite-js') {
          // Direct TFLite in JavaScript
          if (tfliteModel.state !== 'loaded') return
          
          const resized = resize(frame, {
            scale: { width: 416, height: 416 },
            pixelFormat: 'rgb',
            dataType: 'float32',
          })
          
          const outputs = tfliteModel.model.runSync([resized])
          detectionResults = processYoloOutput(outputs[0], 416)
          
        } else if (mode.startsWith('onnx-')) {
          // ONNX with different sizes
          if (onnxModel.state !== 'loaded') return
          
          // For now, only 416 is available
          const size = 416 // parseInt(mode.split('-')[1])
          const resized = resize(frame, {
            scale: { width: size, height: size },
            pixelFormat: 'rgb',
            dataType: 'float32',
          })
          
          const outputs = onnxModel.model.runSync([resized])
          console.log('ONNX output type:', typeof outputs[0], 'keys:', Object.keys(outputs[0] || {}))
          detectionResults = processYoloOutput(outputs[0], size)
          
        } else if (mode.startsWith('native-')) {
          // Native C++ mode
          const result = universalScanner(frame, {
            enabledTypes: ['qr_barcode', 'container_h', 'container_v', 'license_plate', 'seal'],
            useTflite: mode === 'native-tflite', // This will crash!
          })
          
          if (result?.results) {
            detectionResults = result.results.map(r => ({
              className: r.type,
              confidence: r.confidence,
              bbox: { x: r.bbox.x, y: r.bbox.y, w: r.bbox.width, h: r.bbox.height }
            }))
          }
        }
        
        const inferenceEnd = Date.now() - inferenceStart.current
        updateDetections(detectionResults, inferenceEnd)
        
      } catch (error) {
        console.error('Frame processing error:', error instanceof Error ? error.message : String(error))
      }
    },
    [mode, tfliteModel.state, onnxModel.state]
  )
  
  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])
  
  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.text}>Requesting Camera Permission...</Text>
      </View>
    )
  }
  
  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>No Camera Device available</Text>
      </View>
    )
  }
  
  const getModeInfo = () => {
    switch (mode) {
      case 'onnx-640': return 'ONNX 640×640'
      case 'onnx-416': return 'ONNX 416×416'
      case 'onnx-320': return 'ONNX 320×320'
      case 'tflite-js': return 'TFLite (JS)'
      case 'native-onnx': return 'Native ONNX'
      case 'native-tflite': return 'Native TFLite (⚠️)'
    }
  }
  
  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />
      
      {/* Render bounding boxes */}
      {detections.map((detection, index) => {
        const modelSize = mode.includes('640') ? 640 : mode.includes('320') ? 320 : 416
        const scaleX = screenW / modelSize
        const scaleY = screenH / modelSize
        
        return (
          <View
            key={index}
            style={[
              styles.boundingBox,
              {
                left: detection.bbox.x * scaleX,
                top: detection.bbox.y * scaleY,
                width: detection.bbox.w * scaleX,
                height: detection.bbox.h * scaleY,
              }
            ]}
          >
            <Text style={styles.detectionText}>
              {detection.className} ({(detection.confidence * 100).toFixed(0)}%)
            </Text>
          </View>
        )
      })}
      
      {/* Info panel */}
      <View style={styles.infoPanel}>
        <Text style={styles.infoTitle}>Performance Monitor</Text>
        <Text style={styles.infoText}>Mode: {getModeInfo()}</Text>
        <Text style={styles.infoText}>FPS: {fps}</Text>
        <Text style={styles.infoText}>Inference: {inferenceTime}ms</Text>
        <Text style={styles.infoText}>Detections: {detections.length}</Text>
        {detections.length > 0 && (
          <Text style={styles.detectionInfo}>
            {detections[0].className} ({(detections[0].confidence * 100).toFixed(0)}%)
          </Text>
        )}
      </View>
      
      {/* Mode selector */}
      <ScrollView style={styles.modeSelector} horizontal showsHorizontalScrollIndicator={false}>
        <TouchableOpacity 
          style={[styles.modeButton, mode === 'native-onnx' && styles.modeButtonActive]}
          onPress={() => setMode('native-onnx')}
        >
          <Text style={styles.modeButtonText}>Native ONNX</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, mode === 'onnx-640' && styles.modeButtonActive]}
          onPress={() => setMode('onnx-640')}
        >
          <Text style={styles.modeButtonText}>ONNX 640</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, mode === 'onnx-416' && styles.modeButtonActive]}
          onPress={() => setMode('onnx-416')}
        >
          <Text style={styles.modeButtonText}>ONNX 416</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, mode === 'onnx-320' && styles.modeButtonActive]}
          onPress={() => setMode('onnx-320')}
        >
          <Text style={styles.modeButtonText}>ONNX 320</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, mode === 'tflite-js' && styles.modeButtonActive]}
          onPress={() => setMode('tflite-js')}
        >
          <Text style={styles.modeButtonText}>TFLite JS</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.modeButton, mode === 'native-tflite' && styles.modeButtonActive]}
          onPress={() => setMode('native-tflite')}
        >
          <Text style={styles.modeButtonText}>TFLite C++ ⚠️</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'black',
  },
  text: {
    fontSize: 18,
    color: 'white',
    marginTop: 10,
  },
  infoPanel: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: 15,
    borderRadius: 10,
  },
  infoTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  infoText: {
    color: 'white',
    fontSize: 14,
  },
  detectionInfo: {
    color: '#00FF00',
    fontSize: 14,
    fontWeight: 'bold',
  },
  modeSelector: {
    position: 'absolute',
    bottom: 30,
    left: 0,
    right: 0,
    height: 60,
    paddingHorizontal: 10,
  },
  modeButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    marginHorizontal: 5,
    height: 40,
    justifyContent: 'center',
  },
  modeButtonActive: {
    backgroundColor: '#007AFF',
  },
  modeButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00FF00',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
  },
  detectionText: {
    position: 'absolute',
    top: -20,
    left: 0,
    color: '#00FF00',
    fontSize: 12,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 5,
    paddingVertical: 2,
  },
})