#!/usr/bin/env python3
"""
Comparative test script for horizontal container OCR inference (PT vs ONNX)
===========================================================================

This script takes a single horizontal container image, resizes it to 320x320 
with padding, runs inference using both PyTorch and ONNX models, and compares
the results.

Usage:
    python test_horizontal_container_compare.py --image path/to/image.jpg --pt-model path/to/model.pt --onnx-model path/to/model.onnx
"""

import cv2
import numpy as np
import argparse
import onnxruntime as ort
from pathlib import Path
import sys
import time
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


def preprocess_image_onnx(image):
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


def run_pt_inference(image_path, model_path, confidence_threshold=0.3):
    """Run inference using PyTorch model"""
    # Load model
    model = YOLO(model_path)
    
    # Load and preprocess image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None
    
    # Time the inference
    start_time = time.time()
    
    # Run inference with specific image size
    results = model(original_image, imgsz=320, conf=confidence_threshold)
    
    inference_time = time.time() - start_time
    
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
    
    return detections, original_image, inference_time


def run_onnx_inference(image_path, model_path, confidence_threshold=0.3):
    """Run inference using ONNX model"""
    # Load image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None
    
    # Letterbox resize
    input_image, scale, (pad_w, pad_h) = letterbox_image(original_image, (320, 320))
    
    # Preprocess
    preprocessed = preprocess_image_onnx(input_image)
    
    # Load ONNX model
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    
    # Time the inference
    start_time = time.time()
    
    # Run inference
    outputs = session.run(None, {input_name: preprocessed})
    
    inference_time = time.time() - start_time
    
    predictions = outputs[0]
    
    # Process predictions
    detections = []
    
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Remove batch dimension
    
    num_predictions = predictions.shape[0]
    num_features = predictions.shape[1]
    num_classes = num_features - 4  # bbox coords (4) + classes
    
    for i in range(num_predictions):
        # Extract bbox and class scores
        x, y, w, h = predictions[i, :4]
        class_scores = predictions[i, 4:]
        
        # Get best class
        best_class = np.argmax(class_scores)
        confidence = class_scores[best_class]
        
        if confidence < confidence_threshold:
            continue
        
        # Convert from center format to corner format
        x1 = (x - w/2) * 320
        y1 = (y - h/2) * 320
        x2 = (x + w/2) * 320
        y2 = (y + h/2) * 320
        
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
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'class': best_class,
            'confidence': confidence,
            'center_x': (x1 + x2) / 2
        })
    
    # Apply NMS
    if detections:
        boxes = [d['bbox'] for d in detections]
        scores = [d['confidence'] for d in detections]
        keep_indices = non_max_suppression(boxes, scores)
        detections = [detections[i] for i in keep_indices]
    
    # Sort by x-coordinate (left to right)
    detections = sorted(detections, key=lambda d: d['center_x'])
    
    return detections, original_image, inference_time


def visualize_comparison(image, pt_detections, onnx_detections, char_map):
    """Draw both sets of detections on image for comparison"""
    vis_image = image.copy()
    height = vis_image.shape[0]
    
    # Draw PT detections in green (top half)
    for i, det in enumerate(pt_detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        class_idx = det['class']
        confidence = det['confidence']
        
        # Draw bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get character
        char = char_map[class_idx] if class_idx < len(char_map) else '?'
        
        # Draw label
        label = f"PT: {char} {confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ONNX detections in blue (bottom half)
    for i, det in enumerate(onnx_detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        class_idx = det['class']
        confidence = det['confidence']
        
        # Offset slightly to avoid overlap
        y_offset = 20
        
        # Draw bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Get character
        char = char_map[class_idx] if class_idx < len(char_map) else '?'
        
        # Draw label
        label = f"ONNX: {char} {confidence:.2f}"
        cv2.putText(vis_image, label, (x1, y2+y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return vis_image


def compare_detections(pt_detections, onnx_detections, char_map):
    """Compare PT and ONNX detections"""
    pt_string = ''.join([char_map[d['class']] if d['class'] < len(char_map) else '?' for d in pt_detections])
    onnx_string = ''.join([char_map[d['class']] if d['class'] < len(char_map) else '?' for d in onnx_detections])
    
    # Calculate average confidence
    pt_avg_conf = np.mean([d['confidence'] for d in pt_detections]) if pt_detections else 0
    onnx_avg_conf = np.mean([d['confidence'] for d in onnx_detections]) if onnx_detections else 0
    
    # Check if strings match
    strings_match = pt_string == onnx_string
    
    # Calculate position differences for matching characters
    position_diffs = []
    if len(pt_detections) == len(onnx_detections):
        for pt_det, onnx_det in zip(pt_detections, onnx_detections):
            if pt_det['class'] == onnx_det['class']:
                # Calculate center difference
                pt_center = [(pt_det['bbox'][0] + pt_det['bbox'][2]) / 2,
                            (pt_det['bbox'][1] + pt_det['bbox'][3]) / 2]
                onnx_center = [(onnx_det['bbox'][0] + onnx_det['bbox'][2]) / 2,
                              (onnx_det['bbox'][1] + onnx_det['bbox'][3]) / 2]
                
                diff = np.sqrt((pt_center[0] - onnx_center[0])**2 + 
                              (pt_center[1] - onnx_center[1])**2)
                position_diffs.append(diff)
    
    avg_position_diff = np.mean(position_diffs) if position_diffs else -1
    
    return {
        'pt_string': pt_string,
        'onnx_string': onnx_string,
        'strings_match': strings_match,
        'pt_avg_confidence': pt_avg_conf,
        'onnx_avg_confidence': onnx_avg_conf,
        'avg_position_diff': avg_position_diff,
        'pt_count': len(pt_detections),
        'onnx_count': len(onnx_detections)
    }


def main():
    parser = argparse.ArgumentParser(description='Compare PT and ONNX horizontal container OCR')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--pt-model', type=str, help='Path to PyTorch model (.pt)')
    parser.add_argument('--onnx-model', type=str, help='Path to ONNX model')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Path to save comparison visualization')
    
    args = parser.parse_args()
    
    # Check that at least one model is provided
    if not args.pt_model and not args.onnx_model:
        print("Error: At least one model (--pt-model or --onnx-model) must be provided")
        sys.exit(1)
    
    # Check paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Character mapping - must match training data!
    char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    print(f"Processing image: {image_path}")
    print(f"Confidence threshold: {args.confidence}")
    print()
    
    # Run PT inference if model provided
    pt_detections = None
    pt_time = None
    if args.pt_model:
        pt_path = Path(args.pt_model)
        if not pt_path.exists():
            print(f"Error: PT model not found: {pt_path}")
            sys.exit(1)
        
        print(f"Running PyTorch inference with: {pt_path}")
        pt_detections, original_image, pt_time = run_pt_inference(image_path, pt_path, args.confidence)
        
        if pt_detections is not None:
            print(f"PyTorch inference time: {pt_time:.3f}s")
            print(f"PyTorch detected {len(pt_detections)} characters")
            pt_string = ''.join([char_map[d['class']] if d['class'] < len(char_map) else '?' 
                                for d in pt_detections])
            print(f"PyTorch result: {pt_string}")
            
            # Print detailed PT results
            print("\nPyTorch detections (left to right):")
            for i, det in enumerate(pt_detections):
                char = char_map[det['class']] if det['class'] < len(char_map) else '?'
                print(f"  {i+1}. '{char}' - confidence: {det['confidence']:.3f}")
        print()
    
    # Run ONNX inference if model provided
    onnx_detections = None
    onnx_time = None
    if args.onnx_model:
        onnx_path = Path(args.onnx_model)
        if not onnx_path.exists():
            print(f"Error: ONNX model not found: {onnx_path}")
            sys.exit(1)
        
        print(f"Running ONNX inference with: {onnx_path}")
        onnx_detections, original_image, onnx_time = run_onnx_inference(image_path, onnx_path, args.confidence)
        
        if onnx_detections is not None:
            print(f"ONNX inference time: {onnx_time:.3f}s")
            print(f"ONNX detected {len(onnx_detections)} characters")
            onnx_string = ''.join([char_map[d['class']] if d['class'] < len(char_map) else '?' 
                                  for d in onnx_detections])
            print(f"ONNX result: {onnx_string}")
            
            # Print detailed ONNX results
            print("\nONNX detections (left to right):")
            for i, det in enumerate(onnx_detections):
                char = char_map[det['class']] if det['class'] < len(char_map) else '?'
                print(f"  {i+1}. '{char}' - confidence: {det['confidence']:.3f}")
        print()
    
    # Compare results if both models were run
    if pt_detections is not None and onnx_detections is not None:
        print("="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        
        comparison = compare_detections(pt_detections, onnx_detections, char_map)
        
        print(f"PyTorch string:  {comparison['pt_string']} ({comparison['pt_count']} chars)")
        print(f"ONNX string:     {comparison['onnx_string']} ({comparison['onnx_count']} chars)")
        print(f"Strings match:   {comparison['strings_match']}")
        print(f"\nAverage confidence:")
        print(f"  PyTorch: {comparison['pt_avg_confidence']:.3f}")
        print(f"  ONNX:    {comparison['onnx_avg_confidence']:.3f}")
        
        if comparison['avg_position_diff'] >= 0:
            print(f"\nAverage position difference: {comparison['avg_position_diff']:.1f} pixels")
        
        print(f"\nInference speed:")
        print(f"  PyTorch: {pt_time:.3f}s ({1/pt_time:.1f} FPS)")
        print(f"  ONNX:    {onnx_time:.3f}s ({1/onnx_time:.1f} FPS)")
        print(f"  Speedup: {pt_time/onnx_time:.2f}x")
        
        # Save comparison visualization
        if args.output:
            vis_image = visualize_comparison(original_image, pt_detections, onnx_detections, char_map)
            cv2.imwrite(args.output, vis_image)
            print(f"\nComparison visualization saved to: {args.output}")


if __name__ == "__main__":
    main()