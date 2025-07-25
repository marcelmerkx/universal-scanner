project(VisionCameraTflite)
cmake_minimum_required(VERSION 3.9.0)

set (CMAKE_VERBOSE_MAKEFILE ON)
set (PACKAGE_NAME "VisionCameraTflite")
set (CMAKE_CXX_STANDARD 17)

find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)

find_library(
  TFLITE
  tensorflowlite_jni
  PATHS "./src/main/cpp/lib/litert/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(
  TFLITE_GPU
  tensorflowlite_gpu_jni
  PATHS "./src/main/cpp/lib/litert/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

# ONNX Runtime library
find_library(
  ONNXRUNTIME
  onnxruntime
  PATHS "./src/main/cpp/lib/onnxruntime/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

# Note: Using existing image processing utilities instead of OpenCV

string(APPEND CMAKE_CXX_FLAGS " -DANDROID")

add_library(
  ${PACKAGE_NAME}
  SHARED
  src/main/cpp/JniOnLoad.cpp
  ../cpp/jsi/Promise.cpp
  ../cpp/jsi/TypedArray.cpp
  ../cpp/TensorflowPlugin.cpp
  ../cpp/TensorHelpers.cpp
  ../cpp/OnnxPlugin.cpp
  ../cpp/OnnxHelpers.cpp
  ../cpp/preprocessing/FrameConverter.cpp
  ../cpp/preprocessing/ImageRotation.cpp
  ../cpp/preprocessing/WhitePadding.cpp
  src/main/cpp/Tflite.cpp
  src/main/cpp/Onnx.cpp
)

# Specifies a path to native header files.
target_include_directories(
  ${PACKAGE_NAME}
  PRIVATE
  "../cpp"
  "src/main/cpp"
  "src/main/cpp/lib/litert/headers"
  "src/main/cpp/lib/onnxruntime/headers"
  "${NODE_MODULES_DIR}/react-native/ReactCommon"
  "${NODE_MODULES_DIR}/react-native/ReactCommon/callinvoker"
  "${NODE_MODULES_DIR}/react-native/ReactAndroid/src/main/jni/react/turbomodule" # <-- CallInvokerHolder JNI wrapper
)

set_target_properties(${PACKAGE_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
  ${PACKAGE_NAME}
  android                         # <-- log
  log                             # <-- __android_log_print
  ReactAndroid::jsi               # <-- jsi.h
  fbjni::fbjni                    # <-- fbjni.h
  ${TFLITE}
  ${TFLITE_GPU}
  ${ONNXRUNTIME}                  # <-- ONNX Runtime
)

if(ReactAndroid_VERSION_MINOR GREATER_EQUAL 76)
  target_link_libraries(
    ${PACKAGE_NAME}
    ReactAndroid::reactnative     # <-- RN merged so
  )
else()
  target_link_libraries(
    ${PACKAGE_NAME}
    ReactAndroid::reactnativejni  # <-- CallInvokerImpl
  )
endif()

# Universal Scanner native module
add_library(
  universal
  SHARED
  ../cpp/Universal.cpp
  ../cpp/OnnxProcessor.cpp
  ../cpp/OnnxProcessorV2.cpp
  ../cpp/CodeDetectionConstants.cpp
  ../cpp/platform/YuvConverter.cpp
  ../cpp/platform/AndroidYuvConverter.cpp
  ../cpp/platform/AndroidYuvResizer.cpp
  ../cpp/preprocessing/FrameConverter.cpp
  ../cpp/preprocessing/ImageRotation.cpp
  ../cpp/preprocessing/WhitePadding.cpp
  ../cpp/preprocessing/ImageDebugger.cpp
  ../cpp/preprocessing/CropExtractor.cpp
  ../cpp/preprocessing/AdaptiveLetterbox.cpp
  ../cpp/ocr/YoloOCREngine.cpp
  ../cpp/ocr/ContainerOCRProcessor.cpp
)

target_include_directories(
  universal
  PRIVATE
  "../cpp"
  "src/main/cpp"
  "src/main/cpp/lib/onnxruntime/headers"
  "src/main/cpp/lib/litert/headers"
  "src/main/cpp/lib/stb"
)

target_compile_definitions(
  universal
  PRIVATE
  DEBUG  # Enable debug images for universal module only
)

target_link_libraries(
  universal
  android
  log
  fbjni::fbjni
  ${ONNXRUNTIME}
  ${TFLITE}
  ${TFLITE_GPU}
)
