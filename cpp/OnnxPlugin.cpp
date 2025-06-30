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
#include "preprocessing/FrameConverter.h"
#include "preprocessing/ImageRotation.h"
#include "preprocessing/WhitePadding.h"
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
  
  // Run real ONNX inference
  std::vector<float> runInference(const std::vector<uint8_t>& inputData) {
    try {
      // Input is HWC format from resize plugin, need to convert to CHW for ONNX
      const size_t height = 640;
      const size_t width = 640;
      const size_t channels = 3;
      
      // Convert uint8 HWC to float CHW and normalize to [0,1]
      std::vector<float> floatInput(inputData.size());
      
      // Debug: Check input data range before normalization
      uint8_t minVal = 255, maxVal = 0;
      uint32_t sum = 0;
      const size_t sampleSize = std::min(size_t(1000), inputData.size());
      for (size_t i = 0; i < sampleSize; i++) {
        uint8_t val = inputData[i];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
        sum += val;
      }
      float avgVal = static_cast<float>(sum) / sampleSize;
      logf("C++ Input BEFORE norm (first %zu): min=%d, max=%d, avg=%.1f", sampleSize, minVal, maxVal, avgVal);
      
      // Transpose from HWC to CHW and normalize
      float minNorm = 1.0f, maxNorm = 0.0f;
      float sumNorm = 0.0f;
      for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            size_t hwcIdx = (h * width + w) * channels + c;  // HWC index
            size_t chwIdx = c * (height * width) + h * width + w;  // CHW index
            float normalized = static_cast<float>(inputData[hwcIdx]) / 255.0f; // Normalize to [0,1]
            floatInput[chwIdx] = normalized;
            
            // Track normalized range for first 1000 pixels
            if (chwIdx < sampleSize) {
              minNorm = std::min(minNorm, normalized);
              maxNorm = std::max(maxNorm, normalized);
              sumNorm += normalized;
            }
          }
        }
      }
      float avgNorm = sumNorm / sampleSize;
      logf("C++ Input AFTER norm (first %zu): min=%.3f, max=%.3f, avg=%.3f", sampleSize, minNorm, maxNorm, avgNorm);
      
      std::vector<float> tensorData = std::move(floatInput);
      
      // Create input tensor
      const char* inputNames[] = {modelInfo.inputName.c_str()};
      const char* outputNames[] = {modelInfo.outputName.c_str()};
      
      auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, tensorData.data(), tensorData.size(),
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
      
      // Debug output
      logf("ONNX inference completed. Output size: %zu", outputSize);
      
      return output;
      
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("ONNX inference failed: " + std::string(e.what()));
    }
  }
  
  // New method: Run inference on raw frame data with full preprocessing
  std::vector<float> runInferenceWithPreprocessing(
    const UniversalScanner::Frame& frame,
    UniversalScanner::PaddingInfo* padInfo = nullptr
  ) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: YUV to RGB conversion
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> rgbData = UniversalScanner::FrameConverter::convertYUVtoRGB(frame);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Step 2: Rotation if needed
    size_t width = frame.width;
    size_t height = frame.height;
    
    if (UniversalScanner::ImageRotation::needsRotation(width, height)) {
      rgbData = UniversalScanner::ImageRotation::rotate90CW(rgbData, width, height);
      std::swap(width, height);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // Step 3: Run inference with padding
    std::vector<float> output = runInference(rgbData);
    auto t4 = std::chrono::high_resolution_clock::now();
    
    // Log performance metrics
    logf("Preprocessing times (ms):");
    logf("  YUV→RGB: %.2f", std::chrono::duration<double, std::milli>(t2 - t1).count());
    logf("  Rotation: %.2f", std::chrono::duration<double, std::milli>(t3 - t2).count());
    logf("  Inference: %.2f", std::chrono::duration<double, std::milli>(t4 - t3).count());
    logf("  Total: %.2f", std::chrono::duration<double, std::milli>(t4 - start).count());
    
    return output;
  }
};
#else
// Fallback mock for non-Android platforms
struct OnnxSession {
  std::string modelPath;
  Provider provider;
  
  struct {
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    std::vector<int64_t> outputShape = {1, 9, 8400};
    std::string inputName = "images";
    std::string outputName = "output0";
  } modelInfo;
  
  OnnxSession(const Buffer& modelData, Provider prov) : provider(prov) {
    log("Mock ONNX Session created (non-Android platform)");
  }
  
  std::vector<float> runInference(const std::vector<uint8_t>& inputData) {
    const size_t outputSize = 9 * 8400;
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

  // Get the input object (could be TypedArray or Frame)
  jsi::Object inputObject = array.getValueAtIndex(runtime, 0).asObject(runtime);
  
  // Check if it's a Frame object (has width, height properties)
  if (inputObject.hasProperty(runtime, "width") && inputObject.hasProperty(runtime, "height")) {
    // Handle Frame object - do native preprocessing
    processFrameInput(runtime, inputObject);
  } else if (isTypedArray(runtime, inputObject)) {
    // Handle TypedArray - legacy path
    TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(inputObject));
    
    // Store input data for processing
    size_t size = inputBuffer.size(runtime);
    _inputData.resize(size);
    
    // Copy data from TypedArray to vector
    auto arrayBuffer = inputBuffer.getBuffer(runtime);
    uint8_t* data = arrayBuffer.data(runtime);
    std::memcpy(_inputData.data(), data, size);
  } else {
    throw jsi::JSError(runtime, "ONNX: Input value must be a TypedArray or Frame object!");
  }
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

void OnnxPlugin::processFrameInput(jsi::Runtime& runtime, jsi::Object& frameObject) {
  // CRITICAL: JSI Frame objects cannot access native buffers directly
  // We need native frame buffer access which requires a different approach
  
  throw jsi::JSError(runtime, 
    "ONNX Native preprocessing requires native frame buffer access. "
    "Use vision-camera-resize-plugin or implement a native frame processor plugin "
    "that can access Android Image/iOS CVPixelBuffer directly from the camera."
  );
}

void OnnxPlugin::run() {
  auto* session = static_cast<OnnxSession*>(_session);
  if (session == nullptr) {
    throw std::runtime_error("ONNX session is null!");
  }
  
  // Run inference with preprocessed data
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