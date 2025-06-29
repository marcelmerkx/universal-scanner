//
//  OnnxPlugin.cpp
//  UniversalScanner
//
//  Created by Claude Code on 28.06.25.
//

#include "OnnxPlugin.h"

#include "OnnxHelpers.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <memory>

#ifdef ANDROID
#include <android/log.h>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#endif

using namespace facebook;
using namespace mrousavy;

namespace {
  void log(const std::string& message) {
    #ifdef ANDROID
      __android_log_print(ANDROID_LOG_INFO, "OnnxPlugin", "%s", message.c_str());
    #else
      NSLog(@"OnnxPlugin: %s", message.c_str());
    #endif
  }
  
  void logf(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    log(std::string(buffer));
  }
}

#ifdef ANDROID
// Real ONNX Runtime session structure
struct OnnxSession {
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::Env> env;
  Ort::MemoryInfo memoryInfo;
  Provider provider;
  
  // YOLOv8n unified detection model info
  struct {
    std::vector<int64_t> inputShape = {1, 3, 640, 640}; // NCHW format
    std::vector<int64_t> outputShape = {1, 9, 8400}; // 9 features (4 bbox + 5 classes) x 8400 anchors
    std::string inputName = "images";
    std::string outputName = "output0";
  } modelInfo;
  
  OnnxSession(const Buffer& modelData, Provider prov) : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), provider(prov) {
    try {
      // Create ONNX Runtime environment
      env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "UniversalScanner");
      
      // Create session options
      Ort::SessionOptions sessionOptions;
      sessionOptions.SetIntraOpNumThreads(1);
      
      // Set provider based on request
      if (provider == Provider::NnApi) {
        // Try to add NNAPI provider for Android (may not be available in all ONNX Runtime builds)
        try {
          sessionOptions.AppendExecutionProvider("NNAPI");
        } catch (const Ort::Exception& e) {
          logf("NNAPI provider not available, falling back to CPU: %s", e.what());
        }
      }
      // CPU provider is always available as fallback
      
      // Create session from memory buffer
      session = std::make_unique<Ort::Session>(*env, modelData.data, modelData.size, sessionOptions);
      
      // Get input/output info
      Ort::AllocatorWithDefaultOptions allocator;
      
      // Input info
      auto inputName = session->GetInputNameAllocated(0, allocator);
      modelInfo.inputName = inputName.get();
      
      auto inputTypeInfo = session->GetInputTypeInfo(0);
      auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
      modelInfo.inputShape = inputTensorInfo.GetShape();
      
      // Output info  
      auto outputName = session->GetOutputNameAllocated(0, allocator);
      modelInfo.outputName = outputName.get();
      
      auto outputTypeInfo = session->GetOutputTypeInfo(0);
      auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
      modelInfo.outputShape = outputTensorInfo.GetShape();
      
      logf("ONNX Model loaded successfully. Input: %s %ldx%ldx%ldx%ld, Output: %s %ldx%ldx%ld", 
           modelInfo.inputName.c_str(),
           modelInfo.inputShape[0], modelInfo.inputShape[1], modelInfo.inputShape[2], modelInfo.inputShape[3],
           modelInfo.outputName.c_str(),
           modelInfo.outputShape[0], modelInfo.outputShape[1], modelInfo.outputShape[2]);
           
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("Failed to create ONNX session: " + std::string(e.what()));
    }
  }
  
  // Run ONNX inference with native white padding (translated from Kotlin)
  std::vector<float> runInference(const std::vector<uint8_t>& inputData) {
    try {
      const size_t channels = 3;
      const size_t targetSize = 640; // Model expects 640x640
      
      // Detect input dimensions from data size (rotated frame)
      const size_t totalPixels = inputData.size() / channels;
      
      // Common rotated dimensions: 640x480, 960x720, 1440x1080, etc.
      size_t inputHeight = 0, inputWidth = 0;
      
      // Determine dimensions (same logic as before but cleaner)
      if (totalPixels == 640 * 480) {
        inputHeight = 640; inputWidth = 480;
      } else if (totalPixels == 960 * 720) {
        inputHeight = 960; inputWidth = 720;
      } else if (totalPixels == 1440 * 1080) {
        inputHeight = 1440; inputWidth = 1080;
      } else {
        // Fallback: assume 4:3 ratio after rotation
        inputHeight = static_cast<size_t>(std::sqrt(totalPixels * 4.0 / 3.0));
        inputWidth = totalPixels / inputHeight;
      }
      
      logf("Native padding: input %zux%zu (%zu pixels)", inputWidth, inputHeight, totalPixels);
      
      // Calculate aspect-ratio preserving scale (same as Kotlin)
      float scale = std::min(static_cast<float>(targetSize) / inputWidth,
                            static_cast<float>(targetSize) / inputHeight);
      size_t scaledWidth = static_cast<size_t>(inputWidth * scale);
      size_t scaledHeight = static_cast<size_t>(inputHeight * scale);
      
      logf("Scale factor: %.3f, scaled to %zux%zu", scale, scaledWidth, scaledHeight);
      logf("Padding: right=%zu, bottom=%zu", targetSize - scaledWidth, targetSize - scaledHeight);
      
      // Create white-filled tensor [C, H, W] (same as Kotlin white canvas)
      std::vector<float> floatInput(channels * targetSize * targetSize, 1.0f);
      
      // Copy scaled image data to top-left (same as Kotlin drawBitmap at 0,0)
      for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < scaledHeight; ++y) {
          for (size_t x = 0; x < scaledWidth; ++x) {
            // Source pixel coordinates with bilinear-style mapping
            size_t srcY = static_cast<size_t>(y / scale);
            size_t srcX = static_cast<size_t>(x / scale);
            
            // Clamp to input bounds
            srcY = std::min(srcY, inputHeight - 1);
            srcX = std::min(srcX, inputWidth - 1);
            
            // Source: HWC format
            size_t srcIdx = (srcY * inputWidth + srcX) * channels + c;
            
            // Destination: CHW format, top-left aligned
            size_t dstIdx = c * (targetSize * targetSize) + y * targetSize + x;
            
            // Normalize and copy (same as Kotlin /255f)
            floatInput[dstIdx] = static_cast<float>(inputData[srcIdx]) / 255.0f;
          }
        }
      }
      
      // Create input tensor
      const char* inputNames[] = {modelInfo.inputName.c_str()};
      const char* outputNames[] = {modelInfo.outputName.c_str()};
      
      auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, floatInput.data(), floatInput.size(),
        modelInfo.inputShape.data(), modelInfo.inputShape.size());
      
      // Run inference
      auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
      
      // Extract output data
      auto& outputTensor = outputTensors[0];
      auto* outputData = outputTensor.GetTensorMutableData<float>();
      auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
      auto outputShape = outputInfo.GetShape();
      
      size_t outputSize = 1;
      for (auto dim : outputShape) {
        outputSize *= dim;
      }
      
      std::vector<float> output(outputData, outputData + outputSize);
      
      // Debug: Log first few raw output values
      logf("ONNX inference completed. Output size: %zu", outputSize);
      logf("First 10 output values: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
           outputSize > 0 ? output[0] : 0.0f,
           outputSize > 1 ? output[1] : 0.0f,
           outputSize > 2 ? output[2] : 0.0f,
           outputSize > 3 ? output[3] : 0.0f,
           outputSize > 4 ? output[4] : 0.0f,
           outputSize > 5 ? output[5] : 0.0f,
           outputSize > 6 ? output[6] : 0.0f,
           outputSize > 7 ? output[7] : 0.0f,
           outputSize > 8 ? output[8] : 0.0f,
           outputSize > 9 ? output[9] : 0.0f);
      
      // Check if we're getting reasonable bbox values (should be in pixel coordinates 0-640)
      float firstBbox = outputSize > 0 ? output[0] : 0.0f;
      if (firstBbox < 1.0f) {
        logf("WARNING: First bbox value %.4f looks like normalized coordinates!", firstBbox);
        logf("This suggests the model might expect different preprocessing!");
      }
      return output;
      
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("ONNX inference failed: " + std::string(e.what()));
    }
  }
};
#else
// Fallback mock for non-Android platforms
struct OnnxSession {
  std::string modelPath;
  Provider provider;
  
  struct {
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    std::vector<int64_t> outputShape = {1, 16, 8400};
    std::string inputName = "images";
    std::string outputName = "output0";
  } modelInfo;
  
  OnnxSession(const Buffer& modelData, Provider prov) : provider(prov) {
    log("Mock ONNX Session created (non-Android platform)");
  }
  
  std::vector<float> runInference(const std::vector<uint8_t>& inputData) {
    const size_t outputSize = 16 * 8400;
    std::vector<float> output(outputSize, 0.0f);
    return output;
  }
};
#endif

void OnnxPlugin::installToRuntime(jsi::Runtime& runtime,
                                  std::shared_ptr<react::CallInvoker> callInvoker,
                                  FetchURLFunc fetchURL) {

  auto func = jsi::Function::createFromHostFunction(
      runtime, jsi::PropNameID::forAscii(runtime, "__loadOnnxModel"), 1,
      [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
          size_t count) -> jsi::Value {
        auto start = std::chrono::steady_clock::now();
        auto modelPath = arguments[0].asString(runtime).utf8(runtime);

        logf("Loading ONNX Model from \"%s\"...", modelPath.c_str());

        Provider providerType = Provider::Default;
        if (count > 1 && arguments[1].isString()) {
          auto provider = arguments[1].asString(runtime).utf8(runtime);
          if (provider == "coreml") {
            providerType = Provider::CoreML;
          } else if (provider == "gpu") {
            providerType = Provider::GPU;
          } else if (provider == "nnapi") {
            providerType = Provider::NnApi;
          } else {
            providerType = Provider::Default;
          }
        }

        auto promise = Promise::createPromise(runtime, [=, &runtime](
                                                           std::shared_ptr<Promise> promise) {
          std::async(std::launch::async, [=, &runtime]() {
            try {
              // Fetch model from URL (JS bundle)
              Buffer buffer = fetchURL(modelPath);

              // Create real ONNX session
              auto* session = new OnnxSession(buffer, providerType);
              
              if (session == nullptr) {
                callInvoker->invokeAsync([=]() { 
                  promise->reject("Failed to create ONNX session for model \"" + modelPath + "\"!"); 
                });
                return;
              }

              auto plugin = std::make_shared<OnnxPlugin>(session, buffer, providerType, callInvoker);

              callInvoker->invokeAsync([=, &runtime]() {
                auto result = jsi::Object::createFromHostObject(runtime, plugin);
                promise->resolve(std::move(result));
              });

              auto end = std::chrono::steady_clock::now();
              logf("Successfully loaded ONNX Model in %i ms!",
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            } catch (std::exception& error) {
              std::string message = error.what();
              callInvoker->invokeAsync([=]() { promise->reject(message); });
            }
          });
        });
        return promise;
      });

  runtime.global().setProperty(runtime, "__loadOnnxModel", func);
}

OnnxPlugin::OnnxPlugin(void* session, Buffer model, Provider provider,
                       std::shared_ptr<react::CallInvoker> callInvoker)
    : _session(session), _model(model), _provider(provider), _callInvoker(callInvoker) {
  
  if (_session == nullptr) {
    throw std::runtime_error("ONNX session is null!");
  }
  
  log("Successfully created ONNX Plugin with real ONNX session!");
}

OnnxPlugin::~OnnxPlugin() {
  if (_model.data != nullptr) {
    free(_model.data);
    _model.data = nullptr;
    _model.size = 0;
  }
  if (_session != nullptr) {
    delete static_cast<OnnxSession*>(_session);
    _session = nullptr;
  }
}

std::shared_ptr<TypedArrayBase>
OnnxPlugin::getOutputArrayForSession(jsi::Runtime& runtime, size_t index) {
  std::string name = "output_" + std::to_string(index);
  if (_outputBuffers.find(name) == _outputBuffers.end()) {
    // Create output buffer for YOLOv8n model (9 x 8400 floats)
    const size_t outputSize = 9 * 8400;
    auto arrayBuffer = runtime.global()
        .getPropertyAsFunction(runtime, "ArrayBuffer")
        .callAsConstructor(runtime, static_cast<double>(outputSize * sizeof(float)))
        .asObject(runtime);
    
    auto float32Array = runtime.global()
        .getPropertyAsFunction(runtime, "Float32Array")
        .callAsConstructor(runtime, arrayBuffer)
        .asObject(runtime);
        
    _outputBuffers[name] = std::make_shared<TypedArrayBase>(runtime, float32Array);
  }
  return _outputBuffers[name];
}

void OnnxPlugin::copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues) {
  if (!inputValues.isArray(runtime)) {
    throw jsi::JSError(runtime, "ONNX: Input Values must be an array!");
  }

  jsi::Array array = inputValues.asArray(runtime);
  size_t count = array.size(runtime);
  if (count != 1) {
    throw jsi::JSError(runtime, "ONNX: Expected exactly 1 input tensor!");
  }

  // Get the input TypedArray
  jsi::Object inputObject = array.getValueAtIndex(runtime, 0).asObject(runtime);
  
  if (!isTypedArray(runtime, inputObject)) {
    throw jsi::JSError(runtime, "ONNX: Input value is not a TypedArray!");
  }

  TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(inputObject));
  
  // Store input data for processing
  size_t size = inputBuffer.size(runtime);
  _inputData.resize(size);
  
  // Copy data from TypedArray to vector
  auto arrayBuffer = inputBuffer.getBuffer(runtime);
  uint8_t* data = arrayBuffer.data(runtime);
  std::memcpy(_inputData.data(), data, size);
}

jsi::Value OnnxPlugin::copyOutputBuffers(jsi::Runtime& runtime) {
  // Create output array with single tensor
  jsi::Array result(runtime, 1);
  
  // Create output object with proper ONNX-RN format
  jsi::Object outputObject(runtime);
  
  if (!_outputData.empty()) {
    auto* session = static_cast<OnnxSession*>(_session);
    
    // YOLOv8n model outputs [1, 9, 8400]
    // But per ONNX-OUTPUT-FORMAT-DISCOVERY.md, we need nested arrays
    const size_t batch = 1;
    const size_t features = session->modelInfo.outputShape[1]; // 9
    const size_t anchors = session->modelInfo.outputShape[2];  // 8400
    
    // Create 3D nested array structure [batch][features][anchors]
    jsi::Array batch3d(runtime, batch);
    
    for (size_t b = 0; b < batch; b++) {
      jsi::Array features2d(runtime, features);
      
      for (size_t f = 0; f < features; f++) {
        jsi::Array anchors1d(runtime, anchors);
        
        for (size_t a = 0; a < anchors; a++) {
          // Index into flat array: [batch, features, anchors] layout
          size_t idx = b * (features * anchors) + f * anchors + a;
          anchors1d.setValueAtIndex(runtime, a, jsi::Value(_outputData[idx]));
        }
        
        features2d.setValueAtIndex(runtime, f, anchors1d);
      }
      
      batch3d.setValueAtIndex(runtime, b, features2d);
    }
    
    // Set the nested array as the 'value' property (per ONNX-RN format)
    outputObject.setProperty(runtime, "value", batch3d);
    
    // Also set shape info
    jsi::Array shapeArray(runtime, 3);
    shapeArray.setValueAtIndex(runtime, 0, jsi::Value(static_cast<int>(batch)));
    shapeArray.setValueAtIndex(runtime, 1, jsi::Value(static_cast<int>(features)));
    shapeArray.setValueAtIndex(runtime, 2, jsi::Value(static_cast<int>(anchors)));
    outputObject.setProperty(runtime, "shape", shapeArray);
  }
  
  result.setValueAtIndex(runtime, 0, outputObject);
  return result;
}

void OnnxPlugin::run() {
  auto* session = static_cast<OnnxSession*>(_session);
  if (session == nullptr) {
    throw std::runtime_error("ONNX session is null!");
  }
  
  // Run mock inference
  std::vector<float> output = session->runInference(_inputData);
  
  // Store output for later retrieval
  _outputData = std::move(output);
}

jsi::Value OnnxPlugin::get(jsi::Runtime& runtime, const jsi::PropNameID& propNameId) {
  auto propName = propNameId.utf8(runtime);

  if (propName == "runSync") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "runModel"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Value {
          copyInputBuffers(runtime, arguments[0].asObject(runtime));
          this->run();
          return copyOutputBuffers(runtime);
        });
  } else if (propName == "run") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "runModel"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Value {
          copyInputBuffers(runtime, arguments[0].asObject(runtime));
          auto promise =
              Promise::createPromise(runtime, [=, &runtime](std::shared_ptr<Promise> promise) {
                std::async(std::launch::async, [=, &runtime]() {
                  try {
                    this->run();
                    this->_callInvoker->invokeAsync([=, &runtime]() {
                      auto result = this->copyOutputBuffers(runtime);
                      promise->resolve(std::move(result));
                    });
                  } catch (std::exception& error) {
                    promise->reject(error.what());
                  }
                });
              });
          return promise;
        });
  } else if (propName == "inputs") {
    auto* session = static_cast<OnnxSession*>(_session);
    jsi::Array tensors(runtime, 1);
    
    jsi::Object inputTensor = OnnxHelpers::tensorToJSObject(
        runtime, 
        session->modelInfo.inputName, 
        "uint8", 
        session->modelInfo.inputShape
    );
    
    tensors.setValueAtIndex(runtime, 0, inputTensor);
    return tensors;
  } else if (propName == "outputs") {
    auto* session = static_cast<OnnxSession*>(_session);
    jsi::Array tensors(runtime, 1);
    
    jsi::Object outputTensor = OnnxHelpers::tensorToJSObject(
        runtime, 
        session->modelInfo.outputName, 
        "float32", 
        session->modelInfo.outputShape
    );
    
    tensors.setValueAtIndex(runtime, 0, outputTensor);
    return tensors;
  } else if (propName == "provider") {
    switch (_provider) {
      case Provider::Default:
        return jsi::String::createFromUtf8(runtime, "cpu");
      case Provider::CoreML:
        return jsi::String::createFromUtf8(runtime, "coreml");
      case Provider::GPU:
        return jsi::String::createFromUtf8(runtime, "gpu");
      case Provider::NnApi:
        return jsi::String::createFromUtf8(runtime, "nnapi");
    }
  }

  return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> OnnxPlugin::getPropertyNames(jsi::Runtime& runtime) {
  std::vector<jsi::PropNameID> result;
  result.push_back(jsi::PropNameID::forAscii(runtime, "run"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "runSync"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "inputs"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "outputs"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "provider"));
  return result;
}