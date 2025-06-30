package com.cargosnap.app

import android.graphics.*
import android.net.Uri
import android.util.Log
import com.facebook.react.bridge.*
import com.facebook.react.ReactPackage
import com.facebook.react.uimanager.ViewManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.FileOutputStream
import kotlin.math.min
import androidx.exifinterface.media.ExifInterface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File

/**
 * Native helper that resizes an image to fit inside `targetSize × targetSize`
 * while preserving the aspect-ratio and left-aligning the content. The padded
 * output is written to the app cache dir and the absolute path plus metadata
 * are returned to JS. No bitmap crosses the RN bridge – only strings & numbers.
 *
 * JS wrapper lives in `src/services/preprocessing/ImagePaddingService.ts`.
 */
class ImagePaddingModule(private val reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  companion object {
    const val NAME = "ImageStitchingModule" // Keep legacy JS name for drop-in reuse
    private const val TAG = "ImagePaddingModule"
  }

  override fun getName(): String = NAME

  /**
   * Pads `imageUri` to a left-aligned square (default 640×640) and returns
   * `{ uri, scale, padRight, padBottom, originalWidth, originalHeight }`.
   */
  @ReactMethod
  fun padToSquare(imageUri: String, targetSize: Int, promise: Promise) {
    CoroutineScope(Dispatchers.IO).launch {
      try {
        val size = if (targetSize <= 0) 640 else targetSize
        Log.d(TAG, "padToSquare: source=$imageUri → ${size}x$size")

        // Decode bitmap from various URI schemes
        val srcBitmap: Bitmap? = when {
          imageUri.startsWith("file://") -> BitmapFactory.decodeFile(imageUri.removePrefix("file://"))
          imageUri.startsWith("content://") -> reactContext.contentResolver.openInputStream(Uri.parse(imageUri))?.use {
            BitmapFactory.decodeStream(it)
          }
          else -> BitmapFactory.decodeFile(imageUri) // assume raw path
        }

        if (srcBitmap == null) {
          promise.reject("LOAD_ERROR", "Could not decode image at $imageUri")
          return@launch
        }

        // Fix orientation based on EXIF data
        val rotationBitmap = run {
          try {
            val exif = ExifInterface(Uri.parse(imageUri).path ?: imageUri)
            val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            val matrix = Matrix()
            when (orientation) {
              ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
              ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
              ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            }
            if (!matrix.isIdentity) {
              Bitmap.createBitmap(srcBitmap, 0, 0, srcBitmap.width, srcBitmap.height, matrix, true)
            } else srcBitmap
          } catch (e: Exception) {
            Log.w(TAG, "EXIF read failed, using original bitmap", e)
            srcBitmap
          }
        }

        val origW = rotationBitmap.width
        val origH = rotationBitmap.height

        // Aspect-ratio preserving scale so that the longer side matches target
        val scale = min(size.toFloat() / origW, size.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()

        val scaledBmp = if (scale != 1f) Bitmap.createScaledBitmap(rotationBitmap, scaledW, scaledH, true) else rotationBitmap

        // Create square canvas – Ultralytics uses 114 (grey), but white is ok
        val canvasBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        // Use white background to match original training preprocessing
        val canvas = Canvas(canvasBmp).apply { drawColor(Color.WHITE) }
        canvas.drawBitmap(scaledBmp, 0f, 0f, null) // left/top alignment

        // Persist result to cache dir
        val outFile = java.io.File(reactContext.cacheDir, "padded_${System.currentTimeMillis()}.jpg")
        FileOutputStream(outFile).use { stream ->
          canvasBmp.compress(Bitmap.CompressFormat.JPEG, 95, stream)
        }

        val map = Arguments.createMap().apply {
          putString("uri", outFile.absolutePath)
          putDouble("scale", scale.toDouble())
          putInt("padRight", size - scaledW)
          putInt("padBottom", size - scaledH)
          putInt("originalWidth", origW)
          putInt("originalHeight", origH)
        }

        promise.resolve(map)
      } catch (e: Exception) {
        Log.e(TAG, "padToSquare failed", e)
        promise.reject("PAD_ERROR", e)
      }
    }
  }

  // Data class to pass back tensor metadata to other native modules
  data class PadTensorResult(
    val tensorPath: String,
    val scale: Double,
    val padRight: Int,
    val padBottom: Int,
    val originalWidth: Int,
    val originalHeight: Int,
  )

  /** internal utility performing the heavy work and returning metadata */
  private fun performPadAndTensor(imageUri: String, targetSize: Int): PadTensorResult {
    val size = if (targetSize <= 0) 640 else targetSize

    var srcBitmap = BitmapFactory.decodeFile(imageUri.removePrefix("file://"))
      ?: throw Exception("Cannot decode $imageUri")

    // Apply orientation based on EXIF (same logic as padToSquare)
    try {
      val exif = ExifInterface(Uri.parse(imageUri).path ?: imageUri)
      val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
      val matrix = Matrix()
      when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
        ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
        ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
      }
      if (!matrix.isIdentity) {
        srcBitmap = Bitmap.createBitmap(srcBitmap, 0, 0, srcBitmap.width, srcBitmap.height, matrix, true)
      }
    } catch (e: Exception) {
      Log.w(TAG, "EXIF read failed (tensor path)", e)
    }

    val scaleF = min(size.toFloat() / srcBitmap.width, size.toFloat() / srcBitmap.height)
    val scaledW = (srcBitmap.width * scaleF).toInt()
    val scaledH = (srcBitmap.height * scaleF).toInt()
    val scaledBmp = if (scaleF != 1f) Bitmap.createScaledBitmap(srcBitmap, scaledW, scaledH, true) else srcBitmap

    val canvasBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(canvasBmp).apply { drawColor(Color.WHITE) }
    canvas.drawBitmap(scaledBmp, 0f, 0f, null)

    val floats = FloatArray(3 * size * size)
    val area = size * size
    var idx = 0
    for (y in 0 until size) {
      for (x in 0 until size) {
        val p = canvasBmp.getPixel(x, y)
        floats[idx] = (p shr 16 and 0xFF) / 255f
        floats[area + idx] = (p shr 8 and 0xFF) / 255f
        floats[2 * area + idx] = (p and 0xFF) / 255f
        idx++
      }
    }

    val tensorFile = File(reactContext.cacheDir, "tensor_${System.currentTimeMillis()}.bin")
    val bb: ByteBuffer = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(floats)
    tensorFile.outputStream().use { fos ->
      val arr = ByteArray(bb.capacity())
      bb.clear(); bb.get(arr); fos.write(arr)
    }

    return PadTensorResult(
      tensorPath = tensorFile.absolutePath,
      scale = scaleF.toDouble(),
      padRight = size - scaledW,
      padBottom = size - scaledH,
      originalWidth = srcBitmap.width,
      originalHeight = srcBitmap.height,
    )
  }

  /** Async Promise API exposed to JS (keeps existing signature) */
  @ReactMethod
  fun padAndTensor(imageUri: String, targetSize: Int, promise: Promise) {
    CoroutineScope(Dispatchers.IO).launch {
      try {
        val res = performPadAndTensor(imageUri, targetSize)
        val map = Arguments.createMap().apply {
          putString("tensorPath", "file://${res.tensorPath}")
          putDouble("scale", res.scale)
          putInt("padRight", res.padRight)
          putInt("padBottom", res.padBottom)
          putInt("originalWidth", res.originalWidth)
          putInt("originalHeight", res.originalHeight)
        }
        promise.resolve(map)
      } catch (e: Exception) {
        Log.e(TAG, "padAndTensor failed", e)
        promise.reject("PAD_TENSOR_ERROR", e)
      }
    }
  }

  /** Synchronous variant used by other native modules (YoloBridge) */
  fun padAndTensorSync(imageUri: String, targetSize: Int): PadTensorResult =
    performPadAndTensor(imageUri, targetSize)
}

/**
 * Simple ReactPackage that exposes [ImagePaddingModule] to RN.
 */
class ImagePaddingPackage : ReactPackage {
  override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> =
    listOf(ImagePaddingModule(reactContext))

  override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> =
    emptyList()
} 