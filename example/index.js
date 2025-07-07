import { AppRegistry } from 'react-native'
// import App from './src/App' // Original TFLite-only app  
// import App from './src/SwitchableApp' // New app with TFLite/ONNX switcher
// import App from './src/NativePreprocessingApp' // Native universal scanner app
import App from './src/ModelComparisonApp' // Model comparison for FPS optimization
// import App from './src/DirectTfliteApp' // Direct TFLite usage as designed!
// import App from './src/HybridTfliteApp' // Hybrid app with JS TFLite processing
// import App from './src/SimpleTfliteTestApp' // Simple test without camera
// import App from './src/EnhancedSwitchableApp' // All-in-one comparison app
// import App from './src/SimpleSwitchableApp' // Simple working comparison
// import App from './src/SimpleSwitchableAppFixed' // Fixed version based on NativePreprocessingApp
import { name as appName } from './app.json'

AppRegistry.registerComponent(appName, () => App)