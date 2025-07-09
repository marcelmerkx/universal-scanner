#pragma once

#include <vector>
#include <cstdint>

namespace universalscanner {

// Simple image structure to replace cv::Mat
struct ImageData {
    std::vector<uint8_t> data;  // RGB data
    int width;
    int height;
    int channels;  // Usually 3 for RGB
    
    ImageData() : width(0), height(0), channels(3) {}
    
    ImageData(int w, int h, int c = 3) 
        : data(w * h * c), width(w), height(h), channels(c) {}
    
    // Get pixel at (x, y)
    uint8_t* getPixel(int x, int y) {
        return &data[(y * width + x) * channels];
    }
    
    const uint8_t* getPixel(int x, int y) const {
        return &data[(y * width + x) * channels];
    }
    
    // Create a crop of this image
    ImageData crop(int x, int y, int w, int h) const {
        ImageData result(w, h, channels);
        
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                int srcX = x + col;
                int srcY = y + row;
                
                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    const uint8_t* srcPixel = getPixel(srcX, srcY);
                    uint8_t* dstPixel = result.getPixel(col, row);
                    
                    for (int c = 0; c < channels; c++) {
                        dstPixel[c] = srcPixel[c];
                    }
                } else {
                    // Fill with black for out-of-bounds
                    uint8_t* dstPixel = result.getPixel(col, row);
                    for (int c = 0; c < channels; c++) {
                        dstPixel[c] = 0;
                    }
                }
            }
        }
        
        return result;
    }
    
    // Simple bilinear resize
    ImageData resize(int newWidth, int newHeight) const {
        ImageData result(newWidth, newHeight, channels);
        
        float xRatio = static_cast<float>(width) / newWidth;
        float yRatio = static_cast<float>(height) / newHeight;
        
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                float srcX = x * xRatio;
                float srcY = y * yRatio;
                
                int x1 = static_cast<int>(srcX);
                int y1 = static_cast<int>(srcY);
                int x2 = std::min(x1 + 1, width - 1);
                int y2 = std::min(y1 + 1, height - 1);
                
                float xWeight = srcX - x1;
                float yWeight = srcY - y1;
                
                const uint8_t* p11 = getPixel(x1, y1);
                const uint8_t* p12 = getPixel(x1, y2);
                const uint8_t* p21 = getPixel(x2, y1);
                const uint8_t* p22 = getPixel(x2, y2);
                
                uint8_t* dstPixel = result.getPixel(x, y);
                
                for (int c = 0; c < channels; c++) {
                    float val = p11[c] * (1 - xWeight) * (1 - yWeight) +
                               p21[c] * xWeight * (1 - yWeight) +
                               p12[c] * (1 - xWeight) * yWeight +
                               p22[c] * xWeight * yWeight;
                    
                    dstPixel[c] = static_cast<uint8_t>(val);
                }
            }
        }
        
        return result;
    }
    
    // Convert to normalized float tensor (CHW format)
    // TODO: is it normal to have this logic here in the definition of the ImageData?
    std::vector<float> toTensor() const {
        std::vector<float> tensor(channels * width * height);
        
        // Convert to CHW format and normalize
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    const uint8_t* pixel = getPixel(x, y);
                    tensor[c * width * height + y * width + x] = pixel[c] / 255.0f;
                }
            }
        }
        
        return tensor;
    }
};

// Rectangle structure
struct Rectangle {
    int x, y, width, height;
    
    Rectangle() : x(0), y(0), width(0), height(0) {}
    Rectangle(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
};

// Point structure
struct Point2f {
    float x, y;
    Point2f() : x(0), y(0) {}
    Point2f(float x_, float y_) : x(x_), y(y_) {}
};

} // namespace universalscanner