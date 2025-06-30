import * as React from 'react'
import { StyleSheet, View, Text, ActivityIndicator } from 'react-native'
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  Frame,
  VisionCameraProxy,
} from 'react-native-vision-camera'
import { ScannerResult } from 'react-native-fast-tflite'
import { Worklets } from 'react-native-worklets-core'

export default function NativePreprocessingApp(): React.ReactNode {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const [lastResult, setLastResult] = React.useState<any>(null)
  const [fps, setFps] = React.useState(0)
  
  // Initialize VisionCamera plugin
  const universalScanner = VisionCameraProxy.initFrameProcessorPlugin('universalScanner')
  
  React.useEffect(() => {
    console.log('universalScanner plugin initialized:', typeof universalScanner, universalScanner)
  }, [universalScanner])
  
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
      
      try {
        console.log('About to call universalScanner, type:', typeof universalScanner)
        
        if (!universalScanner || typeof universalScanner.call !== 'function') {
          console.log('universalScanner.call is not available:', universalScanner)
          return
        }
        
        // Run the universal scanner with native preprocessing using VisionCamera plugin
        const result = universalScanner.call(frame, {
          enabledTypes: ['code_qr_barcode', 'code_container_h', 'code_container_v'],
          verbose: true,
        })
        
        console.log('VisionCamera universalScanner result:', result)
        
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

  const renderResult = () => {
    if (!lastResult) return null
    
    return (
      <View style={styles.resultContainer}>
        <Text style={styles.resultTitle}>VisionCamera Plugin Result:</Text>
        <Text style={styles.detectionValue}>
          Status: {lastResult.status || lastResult.error || 'Unknown'}
        </Text>
        <Text style={styles.detectionValue}>
          Message: {lastResult.message || lastResult.error || 'No message'}
        </Text>
        {lastResult.results && lastResult.results.map((result: any, index: number) => (
          <View key={index} style={styles.detection}>
            <Text style={styles.detectionType}>{result.type}</Text>
            <Text style={styles.detectionValue}>{result.value}</Text>
            <Text style={styles.detectionConfidence}>
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </Text>
          </View>
        ))}
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
            <Text style={styles.title}>Native Preprocessing Demo</Text>
            <Text style={styles.subtitle}>
              All preprocessing (YUV→RGB, rotation, padding) in C++
            </Text>
            {renderResult()}
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
    bottom: 50,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: 15,
    borderRadius: 10,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 10,
  },
  detection: {
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.2)',
  },
  detectionType: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  detectionValue: {
    fontSize: 16,
    color: 'white',
    marginTop: 2,
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
})