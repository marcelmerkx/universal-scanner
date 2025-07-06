/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import { StyleSheet, View, Text, ActivityIndicator } from 'react-native'
import {
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

function tensorToString(tensor: OnnxTensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`
}
function modelToString(model: OnnxModel): string {
  return (
    `ONNX Model (${model.provider}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

export default function OnnxApp(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')

  // TODO: Replace with actual YOLOv8n-v7 model from /detection/models/
  const model = useOnnxModel(require('../assets/efficientdet.onnx'), 'cpu')
  const actualModel = model.state === 'loaded' ? model.model : undefined

  React.useEffect(() => {
    if (actualModel == null) return
    console.log(`ONNX Model loaded! Shape:\n${modelToString(actualModel)}]`)
  }, [actualModel])

  const { resize } = useResizePlugin()

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      if (actualModel == null) {
        // model is still loading...
        return
      }

      console.log(`Running ONNX inference on ${frame}`)
      const resized = resize(frame, {
        scale: {
          width: 320,
          height: 320,
        },
        pixelFormat: 'rgb',
        dataType: 'uint8',
      })
      
      // CRITICAL: ONNX-RN returns nested arrays, NOT flat Float32Array!
      // See ONNX-OUTPUT-FORMAT-DISCOVERY.md for details
      const result = actualModel.runSync([resized])
      
      // Handle nested array output format properly
      const raw3d = result[0] as unknown as number[][][]
      if (raw3d && raw3d[0]) {
        const preds2d = raw3d[0] // Shape: [9, 8400] or [8400, 9]
        const attributes = 9 // 4 bbox + 5 classes for YOLOv8n model
        const predsAlongLastDim = preds2d[0].length !== attributes
        
        // Get first detection's confidence (class scores start at index 4)
        const confidence = predsAlongLastDim 
          ? Math.max(...preds2d.slice(4, 9).map(arr => arr[0])) // [ATTRIBUTES, N]
          : Math.max(...preds2d[0].slice(4, 9)) // [N, ATTRIBUTES]
          
        console.log('ONNX Confidence: ' + confidence.toFixed(3))
      } else {
        console.log('ONNX Result: Invalid output format')
      }
    },
    [actualModel]
  )

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  console.log(`ONNX Model: ${model.state} (${model.model != null})`)

  return (
    <View style={styles.container}>
      {hasPermission && device != null ? (
        <Camera
          device={device}
          style={StyleSheet.absoluteFill}
          isActive={true}
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
        />
      ) : (
        <Text>No Camera available.</Text>
      )}

      {model.state === 'loading' && (
        <ActivityIndicator size="small" color="white" />
      )}

      {model.state === 'error' && (
        <Text>Failed to load ONNX model! {model.error.message}</Text>
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
})