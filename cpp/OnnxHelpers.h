//
//  OnnxHelpers.h
//  UniversalScanner
//
//  Helper functions for ONNX tensors
//

#pragma once

#include <jsi/jsi.h>
#include "jsi/TypedArray.h"
#include <vector>

namespace mrousavy {

using namespace facebook;

class OnnxHelpers {
public:
  static jsi::Object tensorToJSObject(jsi::Runtime& runtime, const std::string& name, 
                                      const std::string& dataType, 
                                      const std::vector<int64_t>& shape);
  
  static jsi::ArrayBuffer createJSBufferForTensor(jsi::Runtime& runtime, size_t size);
};

} // namespace mrousavy