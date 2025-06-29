#!/bin/bash

# Script to download ONNX Runtime for Android
# Version 1.16.3 is stable and well-tested with React Native

ONNX_VERSION="1.16.3"
ANDROID_DIR="android/src/main/cpp/lib/onnxruntime"

echo "Downloading ONNX Runtime v${ONNX_VERSION} for Android..."

# Create directories
mkdir -p "${ANDROID_DIR}/headers"
mkdir -p "${ANDROID_DIR}/jni/arm64-v8a"
mkdir -p "${ANDROID_DIR}/jni/armeabi-v7a"
mkdir -p "${ANDROID_DIR}/jni/x86"
mkdir -p "${ANDROID_DIR}/jni/x86_64"

# Download Android AAR
cd "${ANDROID_DIR}"
wget "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ONNX_VERSION}/onnxruntime-android-${ONNX_VERSION}.aar"

# Extract AAR (it's a zip file)
unzip -o "onnxruntime-android-${ONNX_VERSION}.aar"

# Move JNI libraries to correct locations
mv jni/arm64-v8a/*.so jni/arm64-v8a/
mv jni/armeabi-v7a/*.so jni/armeabi-v7a/
mv jni/x86/*.so jni/x86/
mv jni/x86_64/*.so jni/x86_64/

# Extract headers from the AAR
cd headers
jar xf ../classes.jar
# Move C API headers to accessible location
find . -name "*.h" -exec mv {} . \;

# Clean up
cd ..
rm -f "onnxruntime-android-${ONNX_VERSION}.aar"
rm -f classes.jar
rm -rf META-INF
rm -rf AndroidManifest.xml
rm -rf R.txt
rm -rf public.txt

echo "ONNX Runtime ${ONNX_VERSION} for Android downloaded successfully!"
echo "Libraries are in: ${ANDROID_DIR}/jni/"
echo "Headers are in: ${ANDROID_DIR}/headers/"