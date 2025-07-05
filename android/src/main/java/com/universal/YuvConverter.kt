package com.universal

import android.graphics.*
import android.media.Image
import java.io.ByteArrayOutputStream

/**
 * Android YUV to RGB converter for VisionCamera frames
 * Handles YUV_420_888 format with proper stride and plane handling
 */
class YuvConverter {
    
    /**
     * Convert raw YUV frame data to RGB byte array
     * Handles Android's YUV_420_888 format complexity
     * 
     * @param frameData Raw YUV bytes from VisionCamera
     * @param width Frame width in pixels  
     * @param height Frame height in pixels
     * @return RGB byte array (width * height * 3) or null if conversion fails
     */
    fun convertYuvToRgb(frameData: ByteArray, width: Int, height: Int): ByteArray? {
        return try {
            // Convert raw YUV data to NV21 format that Android can handle
            val nv21Data = convertYuv420ToNv21(frameData, width, height)
            
            // Use Android's YuvImage to convert NV21 to RGB
            val yuvImage = YuvImage(nv21Data, ImageFormat.NV21, width, height, null)
            
            // Compress to JPEG first (Android's efficient path)
            val outputStream = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, outputStream)
            val jpegBytes = outputStream.toByteArray()
            
            // Decode JPEG to Bitmap
            val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                ?: return null
            
            // Extract RGB bytes from bitmap
            val rgbBytes = bitmapToRgbBytes(bitmap)
            
            // Clean up
            bitmap.recycle()
            outputStream.close()
            
            rgbBytes
        } catch (e: Exception) {
            android.util.Log.e("YuvConverter", "YUV to RGB conversion failed", e)
            null
        }
    }
    
    /**
     * Convert YUV_420_888 raw data to NV21 format
     * Assumes the input is already in I420/YUV420p format from C++
     */
    private fun convertYuv420ToNv21(yuv420Data: ByteArray, width: Int, height: Int): ByteArray {
        val ySize = width * height
        val uvSize = ySize / 4
        
        val nv21 = ByteArray(ySize + uvSize * 2)
        
        // Copy Y plane directly
        System.arraycopy(yuv420Data, 0, nv21, 0, ySize)
        
        // Interleave U and V planes for NV21 format
        val uPlaneStart = ySize
        val vPlaneStart = ySize + uvSize
        
        for (i in 0 until uvSize) {
            nv21[ySize + i * 2] = yuv420Data[vPlaneStart + i]     // V first in NV21
            nv21[ySize + i * 2 + 1] = yuv420Data[uPlaneStart + i] // U second in NV21
        }
        
        return nv21
    }
    
    /**
     * Extract RGB bytes from Android Bitmap
     */
    private fun bitmapToRgbBytes(bitmap: Bitmap): ByteArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        val rgbBytes = ByteArray(width * height * 3)
        var rgbIndex = 0
        
        for (pixel in pixels) {
            rgbBytes[rgbIndex++] = ((pixel shr 16) and 0xFF).toByte() // R
            rgbBytes[rgbIndex++] = ((pixel shr 8) and 0xFF).toByte()  // G
            rgbBytes[rgbIndex++] = (pixel and 0xFF).toByte()          // B
        }
        
        return rgbBytes
    }
}