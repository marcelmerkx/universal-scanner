//
//  OnnxPlugin.cpp
//  UniversalScanner
//
//  Created by Claude Code on 28.06.25.
//

#include "OnnxPlugin.h"

#include "TensorHelpers.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>

// TODO: Add ONNX Runtime includes
// #include <onnxruntime_cxx_api.h>

using namespace facebook;
using namespace mrousavy;

void log(std::string string...) {
  // TODO: Figure out how to log to console
}

void OnnxPlugin::installToRuntime(jsi::Runtime& runtime,
                                  std::shared_ptr<react::CallInvoker> callInvoker,
                                  FetchURLFunc fetchURL) {

  auto func = jsi::Function::createFromHostFunction(
      runtime, jsi::PropNameID::forAscii(runtime, "__loadOnnxModel"), 1,
      [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
          size_t count) -> jsi::Value {
        auto start = std::chrono::steady_clock::now();
        auto modelPath = arguments[0].asString(runtime).utf8(runtime);

        log("Loading ONNX Model from \"%s\"...", modelPath.c_str());

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

              // TODO: Load ONNX model using ONNX Runtime
              // For now, create a placeholder session
              void* session = nullptr; // This will be replaced with actual ONNX session
              
              if (session == nullptr) {
                callInvoker->invokeAsync([=]() { 
                  promise->reject("ONNX Runtime not yet implemented - placeholder for model \"" + modelPath + "\"!"); 
                });
                return;
              }

              auto plugin = std::make_shared<OnnxPlugin>(session, buffer, providerType, callInvoker);

              callInvoker->invokeAsync([=, &runtime]() {
                auto result = jsi::Object::createFromHostObject(runtime, plugin);
                promise->resolve(std::move(result));
              });

              auto end = std::chrono::steady_clock::now();
              log("Successfully loaded ONNX Model in %i ms!",
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
    : _session(session), _provider(provider), _model(model), _callInvoker(callInvoker) {
  
  // TODO: Initialize ONNX Runtime session
  
  log("Successfully created ONNX Plugin!");
}

OnnxPlugin::~OnnxPlugin() {
  if (_model.data != nullptr) {
    free(_model.data);
    _model.data = nullptr;
    _model.size = 0;
  }
  if (_session != nullptr) {
    // TODO: Delete ONNX session
    _session = nullptr;
  }
}

std::shared_ptr<TypedArrayBase>
OnnxPlugin::getOutputArrayForSession(jsi::Runtime& runtime, size_t index) {
  std::string name = "output_" + std::to_string(index);
  if (_outputBuffers.find(name) == _outputBuffers.end()) {
    // TODO: Create proper output buffer based on ONNX model output shape
    _outputBuffers[name] = std::make_shared<TypedArrayBase>(TypedArrayKind::Float32Array, 0);
  }
  return _outputBuffers[name];
}

void OnnxPlugin::copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues) {
  // TODO: Implement ONNX input buffer copying
  throw jsi::JSError(runtime, "ONNX input buffer copying not yet implemented!");
}

jsi::Value OnnxPlugin::copyOutputBuffers(jsi::Runtime& runtime) {
  // TODO: Implement ONNX output buffer copying
  throw jsi::JSError(runtime, "ONNX output buffer copying not yet implemented!");
}

void OnnxPlugin::run() {
  // TODO: Implement ONNX model inference
  throw std::runtime_error("ONNX model inference not yet implemented!");
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
    // TODO: Return ONNX input tensor information
    jsi::Array tensors(runtime, 0);
    return tensors;
  } else if (propName == "outputs") {
    // TODO: Return ONNX output tensor information  
    jsi::Array tensors(runtime, 0);
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