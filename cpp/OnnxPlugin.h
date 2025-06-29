//
//  OnnxPlugin.h
//  UniversalScanner
//
//  Created by Claude Code on 28.06.25.
//

#pragma once

#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include <map>
#include <memory>

#include "jsi/TypedArray.h"

namespace mrousavy {

using namespace facebook;

struct Buffer {
  void* data;
  size_t size;
};

typedef std::function<Buffer(std::string)> FetchURLFunc;

enum class Provider {
  Default,
  CoreML,
  GPU,
  NnApi,
};

class OnnxPlugin : public jsi::HostObject {
public:
  explicit OnnxPlugin(void* session, Buffer model, Provider provider,
                      std::shared_ptr<react::CallInvoker> callInvoker);
  ~OnnxPlugin();

  jsi::Value get(jsi::Runtime& runtime, const jsi::PropNameID& propNameId) override;
  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& runtime) override;

  static void installToRuntime(jsi::Runtime& runtime,
                               std::shared_ptr<react::CallInvoker> callInvoker,
                               FetchURLFunc fetchURL);

private:
  void* _session;
  Buffer _model;
  Provider _provider;
  std::shared_ptr<react::CallInvoker> _callInvoker;
  std::map<std::string, std::shared_ptr<TypedArrayBase>> _outputBuffers;
  std::vector<uint8_t> _inputData;
  std::vector<float> _outputData;

  void copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues);
  jsi::Value copyOutputBuffers(jsi::Runtime& runtime);
  void run();
  std::shared_ptr<TypedArrayBase> getOutputArrayForSession(jsi::Runtime& runtime, size_t index);
};

} // namespace mrousavy