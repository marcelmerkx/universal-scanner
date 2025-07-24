#!/usr/bin/env python3
"""
Quick test script for horizontal container OCR inference
========================================================

This script takes a single horizontal container image, resizes it to 320x320 
with padding, runs inference using our OCR model, and outputs all non-overlapping 
detections from left to right.

Usage:
    python test_horizontal_container.py --image path/to/image.jpg --model path/to/model.[onnx|pt]
"""

import cv2
import numpy as np
import argparse
import onnxruntime as ort
from pathlib import Path
import sys
import torch
from ultralytics import YOLO


def letterbox_image(image, target_size=(320, 320)):
    """Resize image with padding to maintain aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # gray padding
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, (pad_w, pad_h)


def preprocess_image(image):
    """Preprocess image for ONNX model"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Apply NMS to remove overlapping boxes"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by confidence score
    indices = np.argsort(scores)[::-1]
    
    selected = []
    while len(indices) > 0:
        # Select box with highest score
        i = indices[0]
        selected.append(i)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[i]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                         (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = box_area + remaining_areas - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with low IoU
        indices = indices[1:][iou < iou_threshold]
    
    return selected


def run_inference(image_path, model_path, confidence_threshold=0.3):
    """Run inference on a single image"""
    # Character mapping - must match training data!
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Load image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Letterbox resize
    input_image, scale, (pad_w, pad_h) = letterbox_image(original_image, (320, 320))
    
    # Preprocess
    preprocessed = preprocess_image(input_image)
    
    # Load ONNX model
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: preprocessed})
    predictions = outputs[0]
    
    # Process predictions
    # YOLOv8 output can be in two formats:
    # [1, 40, N] - attributes first (40 = 4 bbox + 36 classes)
    # [1, N, 40] - anchors first
    detections = []
    
    # Remove batch dimension
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    
    shape = predictions.shape
    num_attributes = 40  # 4 bbox + 36 classes
    num_classes = 36
    
    # Determine layout
    if shape[0] == num_attributes:
        # [40, N] layout - attributes first
        attributes_first = True
        num_anchors = shape[1]
    elif shape[1] == num_attributes:
        # [N, 40] layout - anchors first
        attributes_first = False
        num_anchors = shape[0]
    else:
        print(f"Error: Unexpected output shape {shape}")
        return detections, original_image
    
    # Process each anchor
    for i in range(num_anchors):
        # Get max class probability and also track top 3 for debugging
        class_probs = []
        
        for c in range(num_classes):
            if attributes_first:
                prob = predictions[4 + c, i]  # [40, N] layout
            else:
                prob = predictions[i, 4 + c]  # [N, 40] layout
            class_probs.append((prob, c))
        
        # Sort by probability
        class_probs.sort(reverse=True)
        max_prob = class_probs[0][0]
        max_class = class_probs[0][1]
        
        if max_prob < confidence_threshold:
            continue
        
        # Extract bbox coordinates
        if attributes_first:
            x = predictions[0, i]  # x-center
            y = predictions[1, i]  # y-center
            w = predictions[2, i]  # width
            h = predictions[3, i]  # height
        else:
            x = predictions[i, 0]
            y = predictions[i, 1]
            w = predictions[i, 2]
            h = predictions[i, 3]
        
        # Check if coordinates are normalized (0-1) or pixel space (0-320)
        # If max coordinate is <= 1, assume normalized
        if max(x, y, w, h) <= 1.0:
            # Convert from normalized to pixel coordinates
            x = x * 320
            y = y * 320
            w = w * 320
            h = h * 320
        
        # Convert from center format to corner format
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        # Adjust for padding
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to image bounds
        h_orig, w_orig = original_image.shape[:2]
        x1 = max(0, min(x1, w_orig))
        y1 = max(0, min(y1, h_orig))
        x2 = max(0, min(x2, w_orig))
        y2 = max(0, min(y2, h_orig))
        
        # Store top 3 predictions for debugging
        top3_chars = []
        for prob, cls in class_probs[:3]:
            char = char_map[cls] if cls < len(char_map) else '?'
            top3_chars.append(f"{char}:{prob:.3f}")
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'class': max_class,
            'confidence': max_prob,
            'center_x': (x1 + x2) / 2,
            'top3': top3_chars  # Debug info
        })
    
    # Apply NMS
    if detections:
        boxes = [d['bbox'] for d in detections]
        scores = [d['confidence'] for d in detections]
        keep_indices = non_max_suppression(boxes, scores)
        detections = [detections[i] for i in keep_indices]
    
    # Sort by x-coordinate (left to right)
    detections = sorted(detections, key=lambda d: d['center_x'])
    
    return detections, original_image


def run_pt_inference(image_path, model_path, confidence_threshold=0.3):
    """Run inference using PyTorch model"""
    # Character mapping - must match training data!
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Run inference with specific image size
    results = model(original_image, imgsz=320, conf=confidence_threshold)
    
    # Process results
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            confidence = boxes.conf[i].cpu().numpy()
            class_id = int(boxes.cls[i].cpu().numpy())
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class': class_id,
                'confidence': float(confidence),
                'center_x': (x1 + x2) / 2
            })
    
    # Sort by x-coordinate (left to right)
    detections = sorted(detections, key=lambda d: d['center_x'])
    
    return detections, original_image


def visualize_results(image, detections):
    """Draw bounding boxes on image"""
    vis_image = image.copy()
    
    # Character mapping - must match training data!
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        class_idx = det['class']
        confidence = det['confidence']
        
        # Draw bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get character
        char = char_map[class_idx] if class_idx < len(char_map) else '?'
        
        # Draw label
        label = f"{char} {confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(description='Test horizontal container OCR')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Check paths
    image_path = Path(args.image)
    model_path = Path(args.model)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Detect model type and run appropriate inference
    print(f"Processing image: {image_path}")
    print(f"Using model: {model_path}")
    
    if model_path.suffix.lower() == '.pt':
        print("Detected PyTorch model")
        detections, original_image = run_pt_inference(image_path, model_path, args.confidence)
    elif model_path.suffix.lower() == '.onnx':
        print("Detected ONNX model")
        detections, original_image = run_inference(image_path, model_path, args.confidence)
    else:
        print(f"Error: Unsupported model format {model_path.suffix}")
        sys.exit(1)
    
    if detections is None or original_image is None:
        sys.exit(1)
    
    # Character mapping - must match training data!
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Print results
    print(f"\nDetected {len(detections)} characters (left to right):")
    detected_string = ""
    for i, det in enumerate(detections):
        char = char_map[det['class']] if det['class'] < len(char_map) else '?'
        detected_string += char
        print(f"  {i+1}. '{char}' - confidence: {det['confidence']:.3f}, "
              f"bbox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
              f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
        if 'top3' in det:
            print(f"      Top 3: {', '.join(det['top3'])}")
    
    print(f"\nDetected container code: {detected_string}")
    
    # Save visualization if requested
    if args.output:
        vis_image = visualize_results(original_image, detections)
        cv2.imwrite(args.output, vis_image)
        print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()