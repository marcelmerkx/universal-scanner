#pragma once

#include <string>
#include <map>
#include <vector>
#include <any>

namespace universalscanner {

struct ScanResult {
    std::string type;           // e.g., "code_container_v"
    std::string value;          // e.g., "MSKU1234567"
    float confidence;           // Overall confidence
    std::string model;          // Model used
    
    struct BoundingBox {
        float x, y, width, height;
    } bbox;
    
    // Image paths (to be set by the frame processor)
    std::string imageCropPath;
    std::string fullFramePath;
    
    // Additional verbose data
    std::map<std::string, std::any> verbose;
};

} // namespace universalscanner