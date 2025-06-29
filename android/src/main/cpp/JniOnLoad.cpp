#include <jni.h>
#include <fbjni/fbjni.h>

namespace mrousavy {
  extern void registerTfliteNatives();
  extern void registerOnnxNatives();
  JavaVM* java_machine = nullptr;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  mrousavy::java_machine = vm;
  return facebook::jni::initialize(vm, [] {
    mrousavy::registerTfliteNatives();
    mrousavy::registerOnnxNatives();
  });
}