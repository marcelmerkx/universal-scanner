//
//  OnnxHelpers.cpp
//  UniversalScanner
//

#include "OnnxHelpers.h"

namespace mrousavy {

jsi::Object OnnxHelpers::tensorToJSObject(jsi::Runtime& runtime, const std::string& name, 
                                          const std::string& dataType, 
                                          const std::vector<int64_t>& shape) {
  jsi::Object object(runtime);
  
  object.setProperty(runtime, "name", jsi::String::createFromUtf8(runtime, name));
  object.setProperty(runtime, "dataType", jsi::String::createFromUtf8(runtime, dataType));
  
  jsi::Array shapeArray(runtime, shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    shapeArray.setValueAtIndex(runtime, i, static_cast<double>(shape[i]));
  }
  object.setProperty(runtime, "shape", shapeArray);
  
  return object;
}

jsi::ArrayBuffer OnnxHelpers::createJSBufferForTensor(jsi::Runtime& runtime, size_t size) {
  auto arrayBufferCtor = runtime.global().getPropertyAsFunction(runtime, "ArrayBuffer");
  auto arrayBufferObj = arrayBufferCtor.callAsConstructor(runtime, static_cast<double>(size));
  return arrayBufferObj.asObject(runtime).getArrayBuffer(runtime);
}

} // namespace mrousavy