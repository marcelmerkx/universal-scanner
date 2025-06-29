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
    std::vector<int64_t> outputShape = {1, 16, 8400}; // 16 features (4 bbox + 1 obj + 11 classes) x 8400 anchors
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
      // Convert uint8 input to float and normalize to [0,1]
      std::vector<float> floatInput(inputData.size());
      for (size_t i = 0; i < inputData.size(); ++i) {
        floatInput[i] = static_cast<float>(inputData[i]) / 255.0f;
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
      
      logf("ONNX inference completed. Output size: %zu", outputSize);
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
    // Create output buffer for YOLOv8n model (16 x 8400 floats)
    const size_t outputSize = 16 * 8400;
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
  
  // Get output buffer and update with inference results
  auto outputBuffer = getOutputArrayForSession(runtime, 0);
  
  // Copy _outputData to the JS buffer
  if (!_outputData.empty()) {
    auto arrayBuffer = outputBuffer->getBuffer(runtime);
    uint8_t* data = arrayBuffer.data(runtime);
    std::memcpy(data, _outputData.data(), _outputData.size() * sizeof(float));
  }
  
  result.setValueAtIndex(runtime, 0, *outputBuffer);
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