/**
 * Code Detection Types for Universal Scanner
 * 
 * These constants match the CodeDetectionConstants.h definitions
 * and are derived from detection/models/unified-detection-v7.yaml
 */

export enum CodeDetectionType {
  CONTAINER_H = 'code_container_h',
  CONTAINER_V = 'code_container_v', 
  LICENSE_PLATE = 'code_license_plate',
  QR_BARCODE = 'code_qr_barcode',
  SEAL = 'code_seal'
}

export const CODE_DETECTION_TYPES = {
  CONTAINER_H: 'code_container_h' as const,
  CONTAINER_V: 'code_container_v' as const,
  LICENSE_PLATE: 'code_license_plate' as const,
  QR_BARCODE: 'code_qr_barcode' as const,
  SEAL: 'code_seal' as const,
} as const;

export type CodeDetectionTypeName = typeof CODE_DETECTION_TYPES[keyof typeof CODE_DETECTION_TYPES];

/**
 * All supported code detection types
 */
export const ALL_CODE_DETECTION_TYPES: CodeDetectionTypeName[] = [
  CODE_DETECTION_TYPES.CONTAINER_H,
  CODE_DETECTION_TYPES.CONTAINER_V,
  CODE_DETECTION_TYPES.LICENSE_PLATE,
  CODE_DETECTION_TYPES.QR_BARCODE,
  CODE_DETECTION_TYPES.SEAL,
];