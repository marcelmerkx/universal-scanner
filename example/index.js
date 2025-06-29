import { AppRegistry } from 'react-native'
// import App from './src/App' // Original TFLite-only app
import App from './src/SwitchableApp' // New app with TFLite/ONNX switcher
import { name as appName } from './app.json'

AppRegistry.registerComponent(appName, () => App)
