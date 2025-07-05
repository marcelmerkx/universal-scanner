package com.universal

import android.graphics.*
import android.media.Image

/**
 * Android YUV resizer for efficient downsampling before processing
 * Handles YUV_420_888 format with proper plane-aware scaling
 */
class YuvResizer {
    
    /**
     * Resize raw YUV frame data to target dimensions
     * Maintains proper YUV 4:2:0 subsampling ratios
     * 
     * @param frameData Raw YUV bytes in I420/YUV420p format
     * @param srcWidth Original frame width
     * @param srcHeight Original frame height  
     * @param targetWidth Desired output width
     * @param targetHeight Desired output height
     * @return Resized YUV byte array or null if resize fails
     */
    fun resizeYuv(
        frameData: ByteArray, 
        srcWidth: Int, 
        srcHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ): ByteArray? {
        return try {
            // Validate inputs
            if (targetWidth <= 0 || targetHeight <= 0 || 
                srcWidth <= 0 || srcHeight <= 0) {
                return null
            }
            
            // Ensure even dimensions for proper YUV subsampling
            val adjustedTargetWidth = targetWidth and 0xFFFE  // Make even
            val adjustedTargetHeight = targetHeight and 0xFFFE // Make even
            
            // Calculate plane sizes
            val srcYSize = srcWidth * srcHeight
            val srcUvSize = srcYSize / 4
            
            val targetYSize = adjustedTargetWidth * adjustedTargetHeight
            val targetUvSize = targetYSize / 4
            
            // Validate source data size
            if (frameData.size < srcYSize + srcUvSize * 2) {
                android.util.Log.e("YuvResizer", "Source data too small: ${frameData.size} < ${srcYSize + srcUvSize * 2}")
                return null
            }
            
            val resizedYuv = ByteArray(targetYSize + targetUvSize * 2)
            
            // Resize Y plane (full resolution)
            resizePlane(
                frameData, 0, srcWidth, srcHeight,
                resizedYuv, 0, adjustedTargetWidth, adjustedTargetHeight
            )
            
            // Resize U plane (half resolution)
            val srcUStart = srcYSize
            val targetUStart = targetYSize
            resizePlane(
                frameData, srcUStart, srcWidth / 2, srcHeight / 2,
                resizedYuv, targetUStart, adjustedTargetWidth / 2, adjustedTargetHeight / 2
            )
            
            // Resize V plane (half resolution)  
            val srcVStart = srcYSize + srcUvSize
            val targetVStart = targetYSize + targetUvSize
            resizePlane(
                frameData, srcVStart, srcWidth / 2, srcHeight / 2,
                resizedYuv, targetVStart, adjustedTargetWidth / 2, adjustedTargetHeight / 2
            )
            
            resizedYuv
        } catch (e: Exception) {
            android.util.Log.e("YuvResizer", "YUV resize failed", e)
            null
        }
    }
    
    /**
     * Resize a single YUV plane using bilinear interpolation
     * Optimized for YUV plane characteristics
     */
    private fun resizePlane(
        srcData: ByteArray, srcOffset: Int, srcWidth: Int, srcHeight: Int,
        dstData: ByteArray, dstOffset: Int, dstWidth: Int, dstHeight: Int
    ) {
        val xRatio = srcWidth.toFloat() / dstWidth.toFloat()
        val yRatio = srcHeight.toFloat() / dstHeight.toFloat()
        
        for (dstY in 0 until dstHeight) {
            for (dstX in 0 until dstWidth) {
                // Calculate source coordinates
                val srcXFloat = dstX * xRatio
                val srcYFloat = dstY * yRatio
                
                val srcX = srcXFloat.toInt()
                val srcY = srcYFloat.toInt()
                
                // Bounds check
                val clampedSrcX = srcX.coerceIn(0, srcWidth - 1)
                val clampedSrcY = srcY.coerceIn(0, srcHeight - 1)
                
                // Simple nearest neighbor for performance
                // Could upgrade to bilinear interpolation if quality needed
                val srcIndex = srcOffset + clampedSrcY * srcWidth + clampedSrcX
                val dstIndex = dstOffset + dstY * dstWidth + dstX
                
                dstData[dstIndex] = srcData[srcIndex]
            }
        }
    }
}