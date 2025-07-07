import { decode } from 'base-64';

// Import the base64 data (we'll create a smaller version)
export function decodeTestImageData(base64Data: string): Float32Array {
  // Decode base64 to binary string
  const binaryString = decode(base64Data);
  
  // Convert binary string to Uint8Array
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  // Convert to Float32Array
  return new Float32Array(bytes.buffer);
}