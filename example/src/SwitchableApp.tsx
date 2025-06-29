/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import { 
  StyleSheet, 
  View, 
  Text, 
  TouchableOpacity,
  ActivityIndicator 
} from 'react-native'
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
  OnnxTensor,
  OnnxModel,
  useOnnxModel,
} from 'react-native-fast-tflite'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera'
import { useResizePlugin } from 'vision-camera-resize-plugin'

type InferenceMode = 'tflite' | 'onnx'

function tensorToString(tensor: Tensor | OnnxTensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`
}

function tfliteModelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

function onnxModelToString(model: OnnxModel): string {
  return (
    `ONNX Model (${model.provider}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

export default function SwitchableApp(): React.ReactNode {
  const [mode, setMode] = React.useState<InferenceMode>('onnx')
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')

  // Load both models
  const tfliteModel = useTensorflowModel(require('../assets/efficientdet.tflite'))
  const onnxModel = useOnnxModel(require('../assets/unified-detection-v7.onnx'), 'cpu')
  
  // Select active model based on mode
  const isModelLoaded = mode === 'tflite' 
    ? tfliteModel.state === 'loaded'
    : onnxModel.state === 'loaded'
    
  const actualModel = mode === 'tflite'
    ? (tfliteModel.state === 'loaded' ? tfliteModel.model : undefined)
    : (onnxModel.state === 'loaded' ? onnxModel.model : undefined)
    
  const modelError = mode === 'tflite'
    ? (tfliteModel.state === 'error' ? tfliteModel.error : undefined)
    : (onnxModel.state === 'error' ? onnxModel.error : undefined)

  React.useEffect(() => {
    if (actualModel == null) return
    if (mode === 'tflite' && tfliteModel.state === 'loaded') {
      console.log(`TFLite Model loaded! Shape:\n${tfliteModelToString(tfliteModel.model)}`)
    } else if (mode === 'onnx' && onnxModel.state === 'loaded') {
      console.log(`ONNX Model loaded! Shape:\n${onnxModelToString(onnxModel.model)}`)
    }
  }, [actualModel, mode, tfliteModel, onnxModel])

  const [frameProcessorsAvailable, setFrameProcessorsAvailable] = React.useState(true)
  const { resize } = useResizePlugin()

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      if (!frameProcessorsAvailable || actualModel == null) {
        // frame processors disabled or model is still loading...
        return
      }

      console.log(`Running ${mode} inference on ${frame}`)
      
      if (mode === 'tflite') {
        const resized = resize(frame, {
          scale: {
            width: 320,
            height: 320,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        })
        const result = actualModel.runSync([resized])
        const num_detections = result[3]?.[0] ?? 0
        console.log('TFLite Result: ' + num_detections)
      } else {
        // ONNX mode - YOLOv8n unified detection
        const resized = resize(frame, {
          scale: {
            width: 640, // YOLOv8 typically uses 640x640
            height: 640,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        })
        
        const result = actualModel.runSync([resized])
        
        // Debug: Log the actual result structure
        console.log('ONNX result length:', result?.length)
        console.log('ONNX result[0] type:', typeof result?.[0])
        console.log('ONNX result[0] keys:', result?.[0] ? Object.keys(result[0]) : 'none')
        
        // ONNX Runtime React Native returns nested arrays, not flat Float32Array
        const output = result[0]
        
        if (output && output.value) {
          // Find max confidence detection - declare outside try block
          let maxConf = 0
          let detectedClass = -1
          let bestBox = { x: 0, y: 0, w: 0, h: 0 }
          
          try {
            // Extract 3D nested array: [batch, features, anchors] or [batch, anchors, features]
            const raw3d = output.value as number[][][]
            const preds2d = raw3d[0] // Remove batch dimension
            
            // YOLOv8n unified detection: 16 features (4 bbox + 1 obj + 11 classes) x 8400 anchors
            const numFeatures = 16
            const numAnchors = 8400
            const numClasses = 11
            
            // Handle both possible orientations: [FEATURES, ANCHORS] or [ANCHORS, FEATURES]
            const featuresAlongFirstDim = preds2d.length === numFeatures
            
            function getVal(anchorIdx: number, featureIdx: number): number {
              return featuresAlongFirstDim
                ? preds2d[featureIdx][anchorIdx]  // [FEATURES, ANCHORS]
                : preds2d[anchorIdx][featureIdx]  // [ANCHORS, FEATURES]
            }
            
            const maxAnchors = featuresAlongFirstDim ? preds2d[0].length : preds2d.length
            const actualAnchors = Math.min(maxAnchors, numAnchors)
            
            // Iterate through all anchors
            for (let anchor = 0; anchor < actualAnchors; anchor++) {
              // Get bounding box
              const centerX = getVal(anchor, 0)
              const centerY = getVal(anchor, 1)
              const width = getVal(anchor, 2)
              const height = getVal(anchor, 3)
              
              // Get objectness score
              const objectness = getVal(anchor, 4)
              
              // Check each class
              for (let c = 0; c < numClasses; c++) {
                const classScore = getVal(anchor, 5 + c)
                const confidence = objectness * classScore
                
                if (confidence > maxConf && confidence > 0.1) { // Basic threshold
                  maxConf = confidence
                  detectedClass = c
                  bestBox = {
                    x: centerX,
                    y: centerY,
                    w: width,
                    h: height
                  }
                }
              }
            }
          } catch (error) {
            console.log('ONNX parsing error:', error)
            return
          }
          
          const classNames = [
            'code_barcode_1d', 'code_qr', 'code_license_plate',
            'code_container_h', 'code_container_v', 'text_printed', 
            'code_seal', 'code_lcd_display', 'code_rail_wagon',
            'code_air_container', 'code_vin'
          ]
          
          if (maxConf > 0.1) {
            console.log(`ONNX Detection: ${classNames[detectedClass] || 'unknown'} (${(maxConf * 100).toFixed(1)}%) at [${bestBox.x.toFixed(0)}, ${bestBox.y.toFixed(0)}, ${bestBox.w.toFixed(0)}, ${bestBox.h.toFixed(0)}]`)
          } else {
            console.log('ONNX: No confident detection')
          }
        } else {
          console.log('ONNX Result: No output data - output:', output ? 'exists but no value' : 'null')
        }
      }
    },
    [actualModel, mode, frameProcessorsAvailable]
  )

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  const toggleMode = () => {
    setMode(current => current === 'tflite' ? 'onnx' : 'tflite')
  }

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <Camera
          device={device}
          style={StyleSheet.absoluteFill}
          isActive={true}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
          onError={(error) => {
            if (error.code === 'system/frame-processors-unavailable') {
              console.log('Frame processors are unavailable')
              setFrameProcessorsAvailable(false)
            } else {
              console.error('Camera error:', error)
            }
          }}
        />
      ) : (
        <Text>No Camera available.</Text>
      )}

      {/* Mode Switcher Button */}
      <TouchableOpacity 
        style={styles.switchButton}
        onPress={toggleMode}
      >
        <Text style={styles.switchText}>
          Mode: {mode.toUpperCase()}
        </Text>
      </TouchableOpacity>

      {/* Loading Indicator */}
      {!isModelLoaded && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="white" />
          <Text style={styles.loadingText}>Loading {mode} model...</Text>
        </View>
      )}

      {/* Error Display */}
      {modelError && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            Failed to load {mode} model! {modelError.message}
          </Text>
        </View>
      )}

      {/* Frame Processors Disabled Warning */}
      {!frameProcessorsAvailable && isModelLoaded && (
        <View style={styles.warningContainer}>
          <Text style={styles.warningText}>
            Frame processors disabled - no inference running
          </Text>
        </View>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  switchButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 10,
    borderRadius: 5,
  },
  switchText: {
    color: 'white',
    fontWeight: 'bold',
  },
  loadingContainer: {
    position: 'absolute',
    bottom: 50,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 10,
    borderRadius: 5,
    flexDirection: 'row',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    marginLeft: 10,
  },
  errorContainer: {
    position: 'absolute',
    bottom: 50,
    backgroundColor: 'rgba(255, 0, 0, 0.7)',
    padding: 10,
    borderRadius: 5,
    maxWidth: '80%',
  },
  errorText: {
    color: 'white',
  },
  warningContainer: {
    position: 'absolute',
    bottom: 100,
    backgroundColor: 'rgba(255, 165, 0, 0.7)',
    padding: 10,
    borderRadius: 5,
    maxWidth: '80%',
  },
  warningText: {
    color: 'white',
  },
})