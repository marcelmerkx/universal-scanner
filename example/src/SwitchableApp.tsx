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
        
        // For our mock ONNX implementation, we get a Float32Array
        const outputData = result[0] as Float32Array
        
        if (outputData && outputData.length > 0) {
          // YOLOv8n output: [1, 16, 8400] -> 16 features x 8400 anchors
          const numAnchors = 8400
          const numFeatures = 16 // 4 bbox + 1 obj + 11 classes
          
          // Find max confidence detection
          let maxConf = 0
          let detectedClass = -1
          let bestBox = { x: 0, y: 0, w: 0, h: 0 }
          
          // Iterate through all anchors
          for (let anchor = 0; anchor < numAnchors; anchor++) {
            const offset = anchor * numFeatures
            
            // Get objectness score
            const objectness = outputData[offset + 4]
            
            // Check each class
            for (let c = 0; c < 11; c++) {
              const classScore = outputData[offset + 5 + c]
              const confidence = objectness * classScore
              
              if (confidence > maxConf) {
                maxConf = confidence
                detectedClass = c
                bestBox = {
                  x: outputData[offset + 0],
                  y: outputData[offset + 1],
                  w: outputData[offset + 2],
                  h: outputData[offset + 3]
                }
              }
            }
          }
          
          const classNames = [
            'code_qr_barcode', 'code_license_plate', 
            'code_container_h', 'code_container_v', 'text_printed',
            'code_seal', 'code_lcd_display', 'code_rail_wagon',
            'code_air_container', 'code_vin'
          ]
          
          console.log(`ONNX Detection: ${classNames[detectedClass] || 'none'} (${(maxConf * 100).toFixed(1)}%)`)
        } else {
          console.log('ONNX Result: Invalid output format')
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