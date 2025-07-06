import React from 'react'
import { View, Text, StyleSheet } from 'react-native'

// Simplified coordinate transformation for camera container approach
export const renderSimplifiedBoundingBoxes = (
  lastResult: any, 
  format: any, 
  screenDimensions: any, 
  previousDetections: Map<string, any>,
  AnimatedBoundingBox: React.FC<any>,
  styles: any
) => {
  if (!lastResult?.detections) return null
  
  // SIMPLIFIED: Camera now fills entire screen (no resizeMode="contain")
  // No need for complex container calculations - use full screen dimensions
  console.log(`🎯 SIMPLIFIED: Camera fills entire screen ${screenDimensions.width}×${screenDimensions.height}`)
  
  return (
    <View style={styles.boundingBoxContainer}>
      {lastResult.detections.map((detection: any, index: number) => {
        // SIMPLIFIED COORDINATE TRANSFORMATION
        const onnxX = parseInt(detection.x)      // Center X in 640x640 ONNX space
        const onnxY = parseInt(detection.y)      // Center Y in 640x640 ONNX space  
        const onnxW = parseInt(detection.width)  // Width in 640x640 ONNX space
        const onnxH = parseInt(detection.height) // Height in 640x640 ONNX space
        
        console.log(`🎯 SIMPLIFIED COORDINATE MAPPING`)
        console.log(`🎯 ONNX: (${onnxX},${onnxY}) size=${onnxW}×${onnxH}`)
        
        // C++ pipeline: 1280×720 → 640×360 → rotate90° → 360×640 → pad → 640×640
        // ONNX coordinates are in 360×640 content space within 640×640 model
        const contentWidth = 360   // Actual content width in ONNX space
        const contentHeight = 640  // Actual content height in ONNX space
        
        // Validate ONNX coordinates are within content area
        if (onnxX > contentWidth || onnxY > contentHeight) {
          console.warn(`⚠️ ONNX coords outside content: (${onnxX},${onnxY}) vs ${contentWidth}×${contentHeight}`)
        }
        
        // Map directly from ONNX content space to camera container
        // ONNX content (360×640) → Container (maintains camera aspect ratio)
        const normalizedX = onnxX / contentWidth    // 0-1 in content width
        const normalizedY = onnxY / contentHeight   // 0-1 in content height
        
        // Map to camera container (which matches exact camera aspect ratio)
        const screenCenterX = normalizedX * screenDimensions.width
        const screenCenterY = normalizedY * screenDimensions.height
        const screenW = (onnxW / contentWidth) * screenDimensions.width
        const screenH = (onnxH / contentHeight) * screenDimensions.height
        
        // Convert from center to top-left for React Native positioning
        const boxLeft = screenCenterX - screenW / 2
        const boxTop = screenCenterY - screenH / 2 // No offset needed - camera fills screen
        
        console.log(`🎯 Normalized: (${normalizedX.toFixed(3)},${normalizedY.toFixed(3)})`)
        console.log(`🎯 Screen: center=(${screenCenterX.toFixed(1)},${screenCenterY.toFixed(1)}) box=(${boxLeft.toFixed(1)},${boxTop.toFixed(1)}) size=${screenW.toFixed(1)}×${screenH.toFixed(1)}`)
        console.log(`🎯 END SIMPLIFIED MAPPING`)
        
        const detectionKey = `${detection.type}-${Math.round(detection.x)}-${Math.round(detection.y)}`
        const previousPosition = previousDetections.get(detectionKey)
        
        return (
          <AnimatedBoundingBox
            key={detectionKey}
            detection={detection}
            boxLeft={Math.max(0, boxLeft)}
            boxTop={Math.max(0, boxTop)}
            screenW={screenW}
            screenH={screenH}
            previousPosition={previousPosition}
          />
        )
      })}
    </View>
  )
}