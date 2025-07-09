/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import { StyleSheet, View, Text, ActivityIndicator } from 'react-native'
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera'
import { useResizePlugin } from 'vision-camera-resize-plugin'

function tensorToString(tensor: Tensor): string {
  return `\n  - ${tensor.dataType} ${tensor.name}[${tensor.shape}]`
}
function modelToString(model: TensorflowModel): string {
  return (
    `TFLite Model (${model.delegate}):\n` +
    `- Inputs: ${model.inputs.map(tensorToString).join('')}\n` +
    `- Outputs: ${model.outputs.map(tensorToString).join('')}`
  )
}

export default function App(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')

  // from https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/tfLite
  // const model = useTensorflowModel(require('../assets/efficientdet.tflite')) // TFLite removed
  // const actualModel = model.state === 'loaded' ? model.model : undefined

  // React.useEffect(() => {
  //   if (actualModel == null) return
  //   console.log(`Model loaded! Shape:\n${modelToString(actualModel)}]`)
  // }, [actualModel])

  const { resize } = useResizePlugin()

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      // TFLite model removed - this app no longer works
      console.log('TFLite model removed - use SwitchableApp or ModelComparisonApp instead')
      return

      console.log(`Running inference on ${frame}`)
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
      console.log('Result: ' + num_detections)
    },
    [actualModel]
  )

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  // console.log(`Model: ${model.state} (${model.model != null})`) // TFLite removed

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

      <View style={{ position: 'absolute', top: 100, left: 20, backgroundColor: 'rgba(255,0,0,0.8)', padding: 10, borderRadius: 5 }}>
        <Text style={{ color: 'white', fontWeight: 'bold' }}>TFLite Removed</Text>
        <Text style={{ color: 'white', fontSize: 12 }}>Use SwitchableApp instead</Text>
      </View>
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
