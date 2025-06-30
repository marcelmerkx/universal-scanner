import type { Frame } from 'react-native-vision-camera'

export interface ScannerConfig {
  /**
   * List of code types to scan for
   */
  enabledTypes?: string[]
  /**
   * Regular expressions per code type for validation
   */
  regexPerType?: Record<string, string[]>
  /**
   * Manual mode - scan only when user taps
   */
  manualMode?: boolean
  /**
   * Enable verbose logging
   */
  verbose?: boolean
}

export interface ScanResult {
  /**
   * Type of code detected
   */
  type: string
  /**
   * Decoded value
   */
  value: string
  /**
   * Confidence score (0-1)
   */
  confidence: number
  /**
   * Bounding box in screen coordinates
   */
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
  /**
   * Path to cropped image of the detection
   */
  imageCropPath?: string
  /**
   * Path to full frame image
   */
  fullFramePath?: string
  /**
   * Model used for detection
   */
  model: string
  /**
   * Additional verbose data
   */
  verbose?: Record<string, any>
}

export interface PaddingInfo {
  scale: number
  scaledWidth: number
  scaledHeight: number
  padLeft: number
  padTop: number
  padRight: number
  padBottom: number
}

export interface ScannerResult {
  results: ScanResult[]
  paddingInfo: PaddingInfo
}

/**
 * Universal Scanner frame processor plugin
 * Processes camera frames to detect and decode various code types
 * 
 * @param frame The camera frame to process
 * @param config Scanner configuration
 * @returns Detection results with padding info for coordinate transformation
 */
export declare function universalScanner(
  frame: Frame,
  config?: ScannerConfig
): ScannerResult | undefined