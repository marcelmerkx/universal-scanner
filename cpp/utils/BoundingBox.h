#pragma once

#include <algorithm>

namespace universalscanner {

// Bounding box structure for detections
struct BoundingBox {
    int x;          // Top-left X coordinate
    int y;          // Top-left Y coordinate  
    int width;      // Width of the box
    int height;     // Height of the box
    
    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
    
    // Check if point is inside the bounding box
    bool contains(int px, int py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
    
    // Calculate area
    int area() const {
        return width * height;
    }
    
    // Calculate intersection with another box
    BoundingBox intersect(const BoundingBox& other) const {
        int x1 = std::max(x, other.x);
        int y1 = std::max(y, other.y);
        int x2 = std::min(x + width, other.x + other.width);
        int y2 = std::min(y + height, other.y + other.height);
        
        if (x1 < x2 && y1 < y2) {
            return BoundingBox(x1, y1, x2 - x1, y2 - y1);
        }
        
        return BoundingBox(); // No intersection
    }
    
    // Calculate IoU (Intersection over Union)
    float iou(const BoundingBox& other) const {
        auto intersection = intersect(other);
        if (intersection.area() == 0) return 0.0f;
        
        int unionArea = area() + other.area() - intersection.area();
        return static_cast<float>(intersection.area()) / unionArea;
    }
};

} // namespace universalscanner