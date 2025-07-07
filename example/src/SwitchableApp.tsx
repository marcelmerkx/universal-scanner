/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import { 
  StyleSheet, 
  View, 
  Text, 
  TouchableOpacity,
  ActivityIndicator,
  Dimensions
} from 'react-native'
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
  OnnxTensor,
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

type InferenceMode = 'tflite' | 'onnx' | 'native' | 'tflite-direct'

interface Detection {
  className: string
  confidence: number
  bbox: { x: number; y: number; w: number; h: number }
}

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
  const [mode, setMode] = React.useState<InferenceMode>('native')
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')

  // Load both models
  const tfliteModel = useTensorflowModel(require('../assets/efficientdet.tflite'))
  const onnxModel = useOnnxModel(require('../assets/unified-detection-v7.onnx'), 'cpu')
  
  // Select active model based on mode
  const isModelLoaded = mode === 'native' 
    ? true // Native mode doesn't need JS model loading - uses shared ONNX session
    : mode === 'tflite' 
      ? tfliteModel.state === 'loaded'
      : onnxModel.state === 'loaded'
    
  const actualModel = mode === 'native'
    ? null // Native mode doesn't use JS model
    : mode === 'tflite'
      ? (tfliteModel.state === 'loaded' ? tfliteModel.model : undefined)
      : (onnxModel.state === 'loaded' ? onnxModel.model : undefined)
    
  const modelError = mode === 'native'
    ? undefined // Native mode doesn't have JS model errors
    : mode === 'tflite'
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
  const [detections, setDetections] = React.useState<Detection[]>([])
  const { resize } = useResizePlugin()

  // Create worklet-compatible function to update detections from the frame processor
  const updateDetectionsJS = React.useMemo(
    () => Worklets.createRunOnJS((newDetections: Detection[]) => {
      setDetections(newDetections)
    }),
    []
  )

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet'
      if (!frameProcessorsAvailable) {
        // frame processors disabled
        return
      }
      
      if (mode !== 'native' && actualModel == null) {
        // model is still loading for tflite/onnx modes
        return
      }

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
      } else if (mode === 'onnx') {
        // ONNX mode - YOLOv8n unified detection 
        // TODO: Native preprocessing requires VisionCamera frame processor plugin
        // Fallback to resize plugin for now
        const resized = resize(frame, {
          scale: {
            width: 640,
            height: 640,
          },
          rotation: '90deg',
          pixelFormat: 'rgb',
          dataType: 'uint8',
        })
        
        const result = actualModel.runSync([resized])
        
        // Native preprocessing completed - detailed logs are in native logcat
        
        // ONNX Runtime React Native returns nested arrays via output.value (per ONNX-OUTPUT-FORMAT-DISCOVERY.md)
        if (result && result[0]) {
          // Find max confidence detection - declare outside try block
          let maxConf = 0
          let detectedClass = -1
          let bestBox = { x: 0, y: 0, w: 0, h: 0 }
          
          // Declare these at the outer scope so they're available later
          let highestScore = 0
          let highestAnchor = -1
          let highestClass = -1
          
          // v7 model has 5 classes (must match training config in detection/models/unified-detection-v7.yaml)
          const classNames = [
                        'code_container_h', 'code_container_v', 'code_license_plate',
            'code_qr_barcode', 'code_seal'
          ]
          
          // Constants for YOLO processing
          const numFeatures = 9
          const numAnchors = 8400
          const numClasses = 5
          
          // Variables that need to be accessible outside try block
          let preds2d: number[][] | null = null
          let featuresAlongFirstDim = false
          let actualAnchors = 0
          let getVal: ((anchorIdx: number, featureIdx: number) => number) | null = null
          
          try {
            // CRITICAL: ONNX-RN returns nested arrays via output.value (as documented)
            const raw3d = result[0].value as number[][][]
            
            // Debug the actual structure
            console.log('ONNX raw3d shape:', raw3d.length, 'x', raw3d[0]?.length, 'x', raw3d[0]?.[0]?.length)
            console.log('ONNX raw3d[0][0][0-4]:', raw3d[0]?.[0]?.slice(0, 5))
            console.log('ONNX raw3d[0][1][0-4]:', raw3d[0]?.[1]?.slice(0, 5))
            console.log('ONNX raw3d[0][4][0-4] (first class):', raw3d[0]?.[4]?.slice(0, 5))
            console.log('ONNX raw3d[0][5][0-4] (second class):', raw3d[0]?.[5]?.slice(0, 5))
            console.log('ONNX raw3d[0][8][0-4] (5th class/last):', raw3d[0]?.[8]?.slice(0, 5))
            
            // Check if data might be transposed
            console.log('Check transposed access:')
            console.log('raw3d[0][0].length:', raw3d[0]?.[0]?.length)
            if (raw3d[0]?.[0]?.length === 9) {
              console.log('Data appears transposed! First anchor all features:', raw3d[0]?.[0])
            }
            
            preds2d = raw3d[0] // Extract 2D array from 3D wrapper: shape [features, anchors] or [anchors, features]
            
            if (!preds2d || !Array.isArray(preds2d) || preds2d.length === 0) {
              console.log('ONNX: Invalid preds2d structure')
              return
            }
            
            // YOLOv8 output format is different! It outputs:
            // [batch_size, num_features, num_anchors] where features = 4 + num_classes
            // BUT the order might be: [x, y, w, h, class1, class2, class3, class4, class5]
            // OR it could be transposed!
            
            console.log('ONNX preds2d shape:', preds2d.length, 'x', preds2d[0]?.length)
            
            // Handle both possible orientations: [FEATURES, ANCHORS] or [ANCHORS, FEATURES]
            featuresAlongFirstDim = preds2d.length === numFeatures
            console.log('Shape analysis: preds2d.length =', preds2d.length, ', numFeatures =', numFeatures, ', featuresAlongFirstDim =', featuresAlongFirstDim)
            
            getVal = (anchorIdx: number, featureIdx: number): number => {
              return featuresAlongFirstDim
                ? preds2d![featureIdx]?.[anchorIdx] ?? 0  // [FEATURES, ANCHORS]
                : preds2d![anchorIdx]?.[featureIdx] ?? 0  // [ANCHORS, FEATURES]
            }
            
            const maxAnchors = featuresAlongFirstDim ? (preds2d[0]?.length ?? 0) : preds2d.length
            actualAnchors = Math.min(maxAnchors, numAnchors)
            
            // Minimal debugging to avoid performance issues
            console.log('featuresAlongFirstDim:', featuresAlongFirstDim)
            if (actualAnchors > 0) {
              const bbox = [getVal!(0, 0), getVal!(0, 1), getVal!(0, 2), getVal!(0, 3)]
              console.log(`Sample anchor 0 bbox:`, bbox.map(v => v.toFixed(2)).join(', '))
            }
            
            // Find which anchors actually have high confidence
            // Scan ALL anchors to find where the 50-67% detections are coming from
            let foundHighConf = false
            
            for (let anchor = 0; anchor < actualAnchors; anchor++) {
              for (let c = 0; c < numClasses; c++) {
                const logit = getVal!(anchor, 4 + c)
                const prob = 1 / (1 + Math.exp(-logit)) // Apply sigmoid
                
                if (prob > highestScore) {
                  highestScore = prob
                  highestAnchor = anchor
                  highestClass = c
                }
                
                // If we find the actual high confidence detection
                if (prob > 0.5 && !foundHighConf) {
                  console.log(`Found HIGH confidence at anchor ${anchor}, class ${c}: logit=${logit.toFixed(4)}, prob=${prob.toFixed(4)}`)
                  // Log all features for this anchor
                  const allFeatures = []
                  for (let f = 0; f < numFeatures; f++) {
                    allFeatures.push(getVal!(anchor, f).toFixed(2))
                  }
                  console.log(`High conf anchor ${anchor} all features:`, allFeatures.join(', '))
                  foundHighConf = true
                }
              }
            }
            
            console.log(`Highest score found: anchor ${highestAnchor}, class ${highestClass} (${classNames[highestClass]}), score=${(highestScore * 100).toFixed(2)}%`)
            
            // Minimal class confidence logging
            if (highestAnchor >= 0 && highestScore > 0.4) {
              console.log(`Best detection: ${classNames[highestClass]} (${(highestScore * 100).toFixed(1)}%)`)
            }
            
            // Debug the highest confidence anchor in detail
            if (highestAnchor >= 0) {
              console.log(`\nAnalyzing highest confidence anchor ${highestAnchor}:`)
              const debugFeatures = []
              for (let f = 0; f < numFeatures; f++) {
                debugFeatures.push(getVal!(highestAnchor, f).toFixed(4))
              }
              console.log(`Anchor ${highestAnchor} all features:`, debugFeatures.join(', '))
            
            // Simple coordinate logging
            const f0 = getVal!(highestAnchor, 0) 
            const f1 = getVal!(highestAnchor, 1) 
            const f2 = getVal!(highestAnchor, 2) 
            const f3 = getVal!(highestAnchor, 3) 
            
            console.log(`Best anchor coords: [${f0.toFixed(1)}, ${f1.toFixed(1)}, ${f2.toFixed(1)}, ${f3.toFixed(1)}]`)
              
              // Convert anchor index to grid position
              // YOLOv8 uses 3 scales with grids: 80x80, 40x40, 20x20 = 6400 + 1600 + 400 = 8400 anchors
              let gridX = 0, gridY = 0, stride = 0
              if (highestAnchor < 6400) {
                // 80x80 grid, stride 8
                const gridIdx = highestAnchor
                gridY = Math.floor(gridIdx / 80)
                gridX = gridIdx % 80
                stride = 8
              } else if (highestAnchor < 8000) {
                // 40x40 grid, stride 16
                const gridIdx = highestAnchor - 6400
                gridY = Math.floor(gridIdx / 40)
                gridX = gridIdx % 40
                stride = 16
              } else {
                // 20x20 grid, stride 32
                const gridIdx = highestAnchor - 8000
                gridY = Math.floor(gridIdx / 20)
                gridX = gridIdx % 20
                stride = 32
              }
              console.log(`Anchor ${highestAnchor} grid position: (${gridX}, ${gridY}) with stride ${stride}`)
              console.log(`Approximate image position: (${gridX * stride}, ${gridY * stride})`)
              
              // Check if this anchor is detecting multiple classes
              console.log(`Anchor ${highestAnchor} class breakdown:`)
              for (let c = 0; c < numClasses; c++) {
                const logit = getVal!(highestAnchor, 4 + c)
                const prob = 1 / (1 + Math.exp(-logit)) // Apply sigmoid
                console.log(`  - ${classNames[c]}: ${(prob * 100).toFixed(2)}% (logit: ${logit.toFixed(3)})`)
              }
              
              // Also check a few anchors around it
              console.log(`\nNearby anchors:`)
              for (let offset = -2; offset <= 2; offset++) {
                if (offset === 0) continue
                const nearbyAnchor = highestAnchor + offset
                if (nearbyAnchor >= 0 && nearbyAnchor < actualAnchors) {
                  const nearbyClass = []
                  for (let c = 0; c < numClasses; c++) {
                    const logit = getVal!(nearbyAnchor, 4 + c)
                    const prob = 1 / (1 + Math.exp(-logit)) // Apply sigmoid
                    nearbyClass.push((prob * 100).toFixed(2) + '%')
                  }
                  console.log(`Anchor ${nearbyAnchor} classes:`, nearbyClass.join(', '))
                }
              }
            }
            
            // Collect all detections for NMS
            interface RawDetection {
              anchor: number
              classId: number
              confidence: number
              bbox: { x: number; y: number; w: number; h: number }
            }
            const allDetections: RawDetection[] = []
            
            // Iterate through all anchors
            let highConfCount = 0
            for (let anchor = 0; anchor < actualAnchors; anchor++) {
              // Get bounding box
              const centerX = getVal(anchor, 0)
              const centerY = getVal(anchor, 1)
              const width = getVal(anchor, 2)   // Back to original order
              const height = getVal(anchor, 3)  // Back to original order
              
              // Get bounding box coordinates (no objectness in v7 model)
              
              // Check each class (classes start at index 4 in v7 model)
              for (let c = 0; c < numClasses; c++) {
                const logit = getVal(anchor, 4 + c)
                const confidence = 1 / (1 + Math.exp(-logit)) // Apply sigmoid
                
                if (confidence > 0.7) { // Raised threshold to 70%
                  highConfCount++
                  if (highConfCount <= 3) {
                    console.log(`High conf detection: anchor ${anchor}, class ${c}, logit ${logit.toFixed(4)}, prob ${confidence.toFixed(4)}`)
                  }
                }
                
                if (confidence > maxConf && confidence > 0.3) { // Use 0.3 threshold for collection
                  maxConf = confidence
                  detectedClass = c
                  bestBox = {
                    x: centerX,
                    y: centerY,
                    w: width,
                    h: height
                  }
                }
                
                // Collect all confident detections
                if (confidence > 0.3) { // Lowered threshold for visualization
                  allDetections.push({
                    anchor: anchor,
                    classId: c,
                    confidence: confidence,
                    bbox: { x: centerX, y: centerY, w: width, h: height }
                  })
                }
              }
            }
            
            // Minimal detection summary to avoid performance issues
            const totalDetections = allDetections.length
            console.log(`Found ${totalDetections} total detections above 0.3 confidence`)
          } catch (error) {
            console.log('ONNX parsing error:', error)
            // Don't return early - let detection logic continue
          }
          
          console.log(`ONNX: Max confidence found: ${(maxConf * 100).toFixed(2)}%`)
          console.log(`ONNX: Best class: ${detectedClass} (${classNames[detectedClass] || 'unknown'})`)
          
          // Based on the detection patterns in logs:
          // - When pointing at container_v, model detects class 3 with 50%+ confidence
          // - Bounding boxes look reasonable (200-300 pixel range)
          // - Model is working but misclassifying
          
          // This could mean:
          // 1. Model was trained with different/insufficient container data
          // 2. The unified-detection-v7.onnx might be a QR-code focused model
          // 3. Container images in training might have looked like QR codes to the model
          
          console.log(`\nDETECTION ANALYSIS:`)
          console.log(`- Detected class: ${detectedClass} (${classNames[detectedClass]})`)
          console.log(`- Confidence: ${(maxConf * 100).toFixed(1)}%`)
          console.log(`- BBox center: (${bestBox.x.toFixed(0)}, ${bestBox.y.toFixed(0)})`)
          console.log(`- BBox size: ${bestBox.w.toFixed(0)}x${bestBox.h.toFixed(0)}`)
          
          // Apply confidence threshold and minimum box size filter
          const MIN_BOX_SIZE = 50 // Minimum width or height in pixels
          const CONFIDENCE_THRESHOLD = 0.3 // 30% confidence threshold for visualization
          
          // When multiple classes have similar high confidence, check which is most likely
          // based on the context (e.g., if user said they're pointing at container_v)
          console.log(`\\nMULTI-CLASS DETECTION CHECK:`)
          const classConfidences: { [key: number]: number } = {}
          if (highestAnchor >= 0) {
            for (let c = 0; c < numClasses; c++) {
              const logit = getVal!(highestAnchor, 4 + c)
              const prob = 1 / (1 + Math.exp(-logit))
              if (prob > 0.7) {
                classConfidences[c] = prob
                console.log(`  ${classNames[c]}: ${(prob * 100).toFixed(1)}%`)
              }
            }
          }
          
          // If multiple classes detected with high confidence, this suggests model confusion
          const highConfClasses = Object.keys(classConfidences).length
          if (highConfClasses > 1) {
            console.log(`  WARNING: Model detected ${highConfClasses} classes with >70% confidence!`)
            console.log(`  This suggests the model may need retraining with better data.`)
          }
          
          console.log(`DETECTION CHECK: maxConf=${(maxConf * 100).toFixed(1)}%, threshold=${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%, boxSize=${bestBox.w.toFixed(0)}x${bestBox.h.toFixed(0)}, minSize=${MIN_BOX_SIZE}`)
          
          if (maxConf > CONFIDENCE_THRESHOLD && (bestBox.w > MIN_BOX_SIZE || bestBox.h > MIN_BOX_SIZE)) {
            console.log(`✅ ONNX Detection PASSED: ${classNames[detectedClass] || 'unknown'} (${(maxConf * 100).toFixed(1)}%) at [${bestBox.x.toFixed(0)}, ${bestBox.y.toFixed(0)}, ${bestBox.w.toFixed(0)}, ${bestBox.h.toFixed(0)}]`)
            
            // Store detection for visualization
            const detection: Detection = {
              className: classNames[detectedClass] || 'unknown',
              confidence: maxConf,
              bbox: bestBox
            }
            console.log(`📦 Creating detection object:`, detection)
            updateDetectionsJS([detection])
          } else {
            console.log(`❌ ONNX Detection FAILED: conf=${(maxConf * 100).toFixed(1)}% (need >${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%), box=${bestBox.w.toFixed(0)}x${bestBox.h.toFixed(0)} (need >${MIN_BOX_SIZE}px)`)
            updateDetectionsJS([])
          }
        } else {
          console.log('ONNX Result: No output.value - result exists:', !!result, 'result[0] exists:', !!result?.[0], 'result[0].value exists:', !!result?.[0]?.value)
        }
      } else if (mode === 'native') {
        // Native mode - Use the UniversalScanner frame processor plugin directly
        // This bypasses all JS overhead and runs everything natively in C++
        
        // First check if universalScanner is available
        if (typeof universalScanner !== 'function') {
          console.log('❌ universalScanner function not available:', typeof universalScanner)
          console.log('Available global functions:', Object.getOwnPropertyNames(global).filter(n => typeof (global as any)[n] === 'function'))
          return
        }
        
        // Also check if ONNX model is loaded (required for native mode)
        if (onnxModel.state !== 'loaded') {
          console.log('❌ Native mode requires ONNX model to be loaded first. Current state:', onnxModel.state)
          return
        }
        
        try {
          const nativeResult = universalScanner(frame, {
            enabledTypes: ['code_qr_barcode', 'code_container_h', 'code_container_v', 'code_license_plate', 'text_printed'],
            verbose: true,
          })
          
          console.log('🚀 Native universalScanner result:', nativeResult)
          
          if (nativeResult && nativeResult.results && nativeResult.results.length > 0) {
            // Convert to our Detection format for visualization
            const detections: Detection[] = nativeResult.results.map((result: any) => ({
              className: result.type,
              confidence: result.confidence,
              bbox: {
                x: result.bbox.x,
                y: result.bbox.y,
                w: result.bbox.width,
                h: result.bbox.height
              }
            }))
            
            console.log(`✅ Native: Found ${detections.length} detections`)
            updateDetectionsJS(detections)
          } else {
            console.log('❌ Native: No detections found')
            updateDetectionsJS([])
          }
        } catch (error) {
          console.log('❌ Native universalScanner error:', error)
          console.log('Error details:', JSON.stringify(error, null, 2))
          if (error instanceof Error) {
            console.log('Error message:', error.message)
            console.log('Error stack:', error.stack)
          }
          updateDetectionsJS([])
        }
      }
    },
    [actualModel, mode, frameProcessorsAvailable, updateDetectionsJS]
  )

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  const toggleMode = () => {
    setMode(current => {
      if (current === 'tflite') return 'onnx'
      if (current === 'onnx') return 'native'
      return 'tflite'
    })
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

      {/* Bounding Box Overlay */}
      {detections.map((detection, index) => {
        // Get screen dimensions
        const screenWidth = Dimensions.get('window').width
        const screenHeight = Dimensions.get('window').height
        
        // vision-camera-resize-plugin coordinate transformation
        // The plugin centers the image in 640x640 space with white padding
        // We need to map from 640x640 model space back to screen space
        
        const modelSize = 640
        
        // Analyze the actual frame dimensions from logs
        // Camera frame: 640x480 (landscape) -> vision-camera-resize-plugin converts to 640x640
        const actualFrameWidth = 640   // From logs: "Camera frame: 640x480"
        const actualFrameHeight = 480
        
        // The plugin preserves aspect ratio, so find the limiting dimension
        const scaleX = modelSize / actualFrameWidth  // 640/640 = 1.0
        const scaleY = modelSize / actualFrameHeight // 640/480 = 1.333
        const paddingScale = Math.min(scaleX, scaleY) // Use 1.0 (limited by X)
        
        const scaledWidth = actualFrameWidth * paddingScale   // 640 * 1.0 = 640
        const scaledHeight = actualFrameHeight * paddingScale // 480 * 1.0 = 480
        
        // The scaled image (640x480) is centered in 640x640 model space
        const offsetX = (modelSize - scaledWidth) / 2   // (640 - 640) / 2 = 0
        const offsetY = (modelSize - scaledHeight) / 2  // (640 - 480) / 2 = 80
        
        console.log(`Actual frame: ${actualFrameWidth}x${actualFrameHeight}`)
        console.log(`Scale factors: X=${scaleX.toFixed(3)}, Y=${scaleY.toFixed(3)}, chosen=${paddingScale.toFixed(3)}`)
        console.log(`Scaled image: ${scaledWidth.toFixed(0)}x${scaledHeight.toFixed(0)} centered in ${modelSize}x${modelSize}`)
        console.log(`Padding offsets: X=${offsetX.toFixed(0)}, Y=${offsetY.toFixed(0)}`)
        
        // Detection at (70, 3) is in 640x640 model space
        // Image occupies Y=80 to Y=560 in this space (centered 480px image)
        // License plate at Y=3 means it's in the padding area? That doesn't make sense.
        // 
        // Actually, let me reconsider: maybe the camera is rotated?
        // If the camera is portrait (480x640), then after rotation it becomes 640x480
        
        // COORDINATE FORMAT DEBUGGING
        // The license plate is in middle of screen but our bbox shows Y=3
        // Let's investigate if we're interpreting the ONNX output format wrong
        
        console.log(`\\nCURRENT INTERPRETATION:`)
        console.log(`bbox.x=${detection.bbox.x.toFixed(1)}, bbox.y=${detection.bbox.y.toFixed(1)}, bbox.w=${detection.bbox.w.toFixed(1)}, bbox.h=${detection.bbox.h.toFixed(1)}`)
        
        // License plate in middle of screen should be around (320, 240) in 640x480 space
        // Or around (320, 320) in 640x640 model space
        // But we're getting (85, 3) - that's way off!
        
        // HYPOTHESIS: Maybe the coordinates from the model are swapped or in different format
        // Let's try swapping width/height with x/y
        const altDetection = {
          x: detection.bbox.w,  // Use width as X (146)
          y: detection.bbox.h,  // Use height as Y (5) - still too small
          w: detection.bbox.x,  // Use X as width (85)
          h: detection.bbox.y   // Use Y as height (3) - way too small
        }
        
        console.log(`\\nALTERNATE INTERPRETATION (swap x/y with w/h):`)
        console.log(`alt.x=${altDetection.x.toFixed(1)}, alt.y=${altDetection.y.toFixed(1)}, alt.w=${altDetection.w.toFixed(1)}, alt.h=${altDetection.h.toFixed(1)}`)
        
        // Maybe the model outputs normalized coordinates (0-1) that need scaling?
        const normalized = {
          x: detection.bbox.x / modelSize * modelSize,  // Already in correct range
          y: detection.bbox.y / modelSize * modelSize,  // Already in correct range  
          w: detection.bbox.w / modelSize * modelSize,
          h: detection.bbox.h / modelSize * modelSize
        }
        
        console.log(`\\nNORMALIZED TEST (x/640*640):`)
        console.log(`norm.x=${normalized.x.toFixed(1)}, norm.y=${normalized.y.toFixed(1)}, norm.w=${normalized.w.toFixed(1)}, norm.h=${normalized.h.toFixed(1)}`)
        
        // For now, use original interpretation but add debug info
        const directScaleX = screenWidth / modelSize  
        const directScaleY = screenHeight / modelSize 
        
        // BACK TO BASICS - use original coordinates to debug step by step
        console.log(`Detection: ${detection.className} at (${detection.bbox.x.toFixed(0)}, ${detection.bbox.y.toFixed(0)}) size ${detection.bbox.w.toFixed(0)}x${detection.bbox.h.toFixed(0)}`)
        
        const left = (detection.bbox.x - detection.bbox.w / 2) * directScaleX
        const top = (detection.bbox.y - detection.bbox.h / 2) * directScaleY
        const width = detection.bbox.w * directScaleX
        const height = detection.bbox.h * directScaleY
        
        console.log(`Screen: (${left.toFixed(0)}, ${top.toFixed(0)}) size ${width.toFixed(0)}x${height.toFixed(0)})`)
        
        return (
          <View
            key={index}
            style={[
              styles.boundingBox,
              {
                left: left,
                top: top,
                width: width,
                height: height,
              },
            ]}
          >
            <View style={styles.labelContainer}>
              <Text 
                style={styles.labelText}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                {detection.className.replace('code_', '')} ({(detection.confidence * 100).toFixed(1)}%)
              </Text>
            </View>
          </View>
        )
      })}

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
  boundingBox: {
    position: 'absolute',
    borderColor: '#00FF00',
    borderWidth: 2,
    borderRadius: 4,
  },
  labelContainer: {
    position: 'absolute',
    top: -25,
    left: 0,
    backgroundColor: '#00FF00',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    alignItems: 'center',
    minWidth: 200,  // Much wider for full class names
    maxWidth: 500,  // Allow very wide labels
  },
  labelText: {
    color: 'black',
    fontSize: 12,
    fontWeight: 'bold',
    textAlign: 'center',
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
})