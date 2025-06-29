//
//  Onnx.mm
//  UniversalScanner
//
//  Created by Claude Code on 28.06.25.
//

#import "Onnx.h"

#import <React/RCTBridge+Private.h>
#import <React/RCTUtils.h>
#import <ReactCommon/RCTTurboModule.h>
#import <jsi/jsi.h>

#import "OnnxPlugin.h"

@implementation Onnx

RCT_EXPORT_MODULE()

- (BOOL)install {
  RCTBridge* bridge = [RCTBridge currentBridge];
  RCTCxxBridge* cxxBridge = (RCTCxxBridge*)bridge;
  if (cxxBridge == nil) {
    return NO;
  }

  auto jsiRuntime = (jsi::Runtime*)cxxBridge.runtime;
  if (jsiRuntime == nil) {
    return NO;
  }
  auto& runtime = *jsiRuntime;

  auto callInvoker = cxxBridge.jsCallInvoker;

  auto fetchURL = [](std::string url) -> mrousavy::Buffer {
    NSString* nsUrl = [NSString stringWithUTF8String:url.c_str()];
    NSURL* requestUrl = [NSURL URLWithString:nsUrl];
    NSData* data = [NSData dataWithContentsOfURL:requestUrl];
    
    void* buffer = malloc(data.length);
    memcpy(buffer, data.bytes, data.length);
    
    return mrousavy::Buffer{
      .data = buffer,
      .size = data.length
    };
  };

  mrousavy::OnnxPlugin::installToRuntime(runtime, callInvoker, fetchURL);
  
  return YES;
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams&)params
{
    return nullptr;
}
#endif

@end