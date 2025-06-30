#pragma once

#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "preprocessing/FrameConverter.h"
#include "preprocessing/WhitePadding.h"

namespace mrousavy {

using namespace facebook;
using namespace UniversalScanner;

// Configuration for the scanner
struct ScannerConfig {
    std::vector<std::string> enabledTypes;
    std::map<std::string, std::vector<std::string>> regexPerType;
    bool manualMode = false;
    bool verbose = false;
};

// Result of a scan
struct ScanResult {
    std::string type;
    std::string value;
    float confidence;
    struct {
        float x, y, width, height;
    } bbox;
    std::string imageCropPath;
    std::string fullFramePath;
    std::string model;
    std::map<std::string, jsi::Value> verbose;
};

class UniversalScannerPlugin {
public:
    // Install the frame processor plugin
    static void install(jsi::Runtime& runtime,
                       std::shared_ptr<react::CallInvoker> callInvoker);
    
    // Process a frame with the universal scanner
    static jsi::Value processFrame(jsi::Runtime& runtime,
                                  const jsi::Value& frame,
                                  const jsi::Value& config);
    
private:
    // Extract scanner configuration from JS object
    static ScannerConfig extractConfig(jsi::Runtime& runtime,
                                       const jsi::Value& configValue);
    
    // Convert scan results to JS object
    static jsi::Value resultsToJSI(jsi::Runtime& runtime,
                                  const std::vector<ScanResult>& results,
                                  const PaddingInfo& padInfo);
    
    // Helper to extract Frame from JS object
    static Frame extractFrame(jsi::Runtime& runtime,
                             const jsi::Value& frameValue);
};

} // namespace mrousavy