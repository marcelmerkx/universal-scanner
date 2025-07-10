/**
 * Container Code Validation Utilities
 * Translated from cpp/ocr/ContainerOCRProcessor.cpp
 */

/**
 * Common OCR confusions: digits that often get misread as letters
 */
function digitToLetter(digit: string): string {
  switch (digit) {
    case '0': return 'O';
    case '1': return 'I';
    case '5': return 'S';
    case '2': return 'Z';
    case '8': return 'B';
    case '6': return 'G';
    default: return digit;
  }
}

/**
 * Inverse mappings: letters that often get misread as digits
 */
function letterToDigit(letter: string): string {
  switch (letter) {
    case 'O': return '0';
    case 'I': return '1';
    case 'S': return '5';
    case 'Z': return '2';
    case 'B': return '8';
    case 'G': return '6';
    default: return letter;
  }
}

/**
 * Calculate ISO 6346 check digit for first 10 characters
 */
function calculateCheckDigit(code: string): number {
  if (code.length !== 10) {
    throw new Error('Code must be exactly 10 characters for check digit calculation');
  }
  
  let sum = 0;
  
  // Letter to number mapping for ISO 6346
  const letterValue = (c: string): number => {
    if (/\d/.test(c)) return parseInt(c, 10);
    
    // A=10, B=12, C=13... (skip multiples of 11)
    let val = (c.charCodeAt(0) - 'A'.charCodeAt(0)) + 10;
    if (val >= 11) val++; // Skip 11
    if (val >= 22) val++; // Skip 22
    if (val >= 33) val++; // Skip 33
    return val;
  };
  
  // Calculate weighted sum
  for (let i = 0; i < 10; i++) {
    sum += letterValue(code[i]) * Math.pow(2, i); // 2^i
  }
  
  return sum % 11;
}

/**
 * Apply ISO 6346 corrections to raw OCR text
 * - First 4 chars must be letters (owner code + equipment category)
 * - Next 6 chars must be digits (serial number)  
 * - Last char is check digit (can be letter or digit)
 */
export function applyISO6346Corrections(raw: string): string {
  console.log(`[ContainerValidation] Applying corrections to: "${raw}"`);
  
  if (raw.length !== 11) {
    console.log(`[ContainerValidation] Wrong length: ${raw.length} (expected 11)`);
    return raw;
  }
  
  let corrected = raw;
  
  // First 4 must be letters (owner code + equipment category)
  for (let i = 0; i < 4; i++) {
    if (/\d/.test(corrected[i])) {
      const original = corrected[i];
      corrected = corrected.substring(0, i) + digitToLetter(corrected[i]) + corrected.substring(i + 1);
      console.log(`[ContainerValidation] Position ${i}: digit "${original}" -> letter "${corrected[i]}"`);
    }
  }
  
  // Next 6 must be digits (serial number)
  for (let i = 4; i < 10; i++) {
    if (/[A-Za-z]/.test(corrected[i])) {
      const original = corrected[i];
      corrected = corrected.substring(0, i) + letterToDigit(corrected[i].toUpperCase()) + corrected.substring(i + 1);
      console.log(`[ContainerValidation] Position ${i}: letter "${original}" -> digit "${corrected[i]}"`);
    }
  }
  
  console.log(`[ContainerValidation] Corrected: "${raw}" -> "${corrected}"`);
  return corrected.toUpperCase();
}

/**
 * Validate ISO 6346 container code format and checksum
 */
export function validateISO6346(code: string): boolean {
  console.log(`[ContainerValidation] Validating ISO 6346: "${code}"`);
  
  if (code.length !== 11) {
    console.log(`[ContainerValidation] Invalid length: ${code.length} (expected 11)`);
    return false;
  }
  
  const upperCode = code.toUpperCase();
  
  // Check format: 4 letters + 6 digits + 1 check digit
  for (let i = 0; i < 4; i++) {
    if (!/[A-Z]/.test(upperCode[i])) {
      console.log(`[ContainerValidation] Position ${i}: "${upperCode[i]}" is not a letter`);
      return false;
    }
  }
  
  for (let i = 4; i < 10; i++) {
    if (!/\d/.test(upperCode[i])) {
      console.log(`[ContainerValidation] Position ${i}: "${upperCode[i]}" is not a digit`);
      return false;
    }
  }
  
  // Calculate and verify check digit
  const calculatedCheck = calculateCheckDigit(upperCode.substring(0, 10));
  const expectedCheck = calculatedCheck === 10 ? 'A' : calculatedCheck.toString();
  const actualCheck = upperCode[10];
  
  const isValid = actualCheck === expectedCheck;
  
  console.log(`[ContainerValidation] Check digit: expected "${expectedCheck}", got "${actualCheck}", valid: ${isValid}`);
  
  return isValid;
}

/**
 * Process and validate container code with corrections
 */
export function processContainerCode(raw: string): { 
  corrected: string; 
  isValid: boolean; 
  original: string 
} {
  console.log(`[ContainerValidation] Processing container code: "${raw}"`);
  
  const corrected = applyISO6346Corrections(raw);
  const isValid = validateISO6346(corrected);
  
  console.log(`[ContainerValidation] Result: original="${raw}", corrected="${corrected}", valid=${isValid}`);
  
  return {
    original: raw,
    corrected,
    isValid
  };
}