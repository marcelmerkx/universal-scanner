# YOLO OCR

## Background
YOLO to also perform the object recognition of the character; so not just the location.
we currently have a pipeline of using YOLO to identify the place of container characters, we then pre-process it to a horizontal code and then feed that to MLKit for OCR. I'd like to make a plan where we skip that all and only YOLO it.

We used the notebook yolo_prototype to download the training set from Roboflow. Lets use that same set, but adjust it for the purpose. Make a new copy of that notebook so we can work on that.

## objective
get to totally shorten the time to results by skipping a bunch of processing. Fast and high accurate: aim higher than MLKit

## Next step
make a plan where we script the replacement of "character" as class in the container codes by the A-Z and 0-9 labels, taken from the filename (top to bottom, replacing the classes).

### Detailed Plan – "YOLO-OCR" multi-class upgrade

1. **✅ Create a working copy of the current dataset**
   - Pull the dataset from Roboflow.
   - Preserve identical folder structure (`images/train`, `images/val`, `labels/train`, `labels/val`).

2. **✅ Define the new class list**
   - Characters: `A-Z` (26) + digits `0-9` (10) = **36 classes**.
   - Generate `data/dataset_yolo_ocr/container_dataset.yaml` with:
     ```yaml
     names: ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
     ```

3. **✅ Write a conversion script `scripts/convert_labels_to_multiclass.py`**
   - Input: path to the *single-class* labels folder, destination folder for *multi-class* labels.
   - Logic:
     1. For each image, sort bounding boxes **top → bottom** by the `y_center` value.
     2. Extract expected container code from the **filename** (e.g., `ABCU1234567.jpg` → `A-B-C-U-1-2-3-4-5-6-7`).
     3. Map every sorted bounding box to the corresponding character in that string.
        ```python
        CHAR_TO_ID: dict[str, int] = {c: idx for idx, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")}
        ```
     4. Replace the leading `0` in each YOLO label line (`<class_id> x y w h`) with the derived `class_id`.
     5. Write the updated lines to the destination `.txt` file **in the same order**.
   - Add exhaustive error checking (length mismatch, unknown characters) and a `--dry-run` flag for safe testing.

4. **✅ Duplicate the training notebook**
   - Copy `notebooks/yolo_prototype.ipynb` → `notebooks/yolo_ocr_train.ipynb`.
   - Update: `data=` path, `nc=36`, `names=` list, and experiment name.
   - Train with:
     ```python
     model = YOLO("yolov8n.pt")
     results = model.train(data="data/dataset_yolo_ocr/container_dataset.yaml", imgsz=640, epochs=100, batch=32, lr0=0.01, patience=20)
     ```

5. **✅ Validation & Metrics**
   - Expect *exactly 11* detections per image (one per character).
   - Use `task=val` in Ultralytics CLI to compute mAP per class.
   - Add a custom validation script that reconstructs the predicted 11-character string and compares it to the ground truth extracted from the filename; log CER / accuracy.

6. **✅ Export for inference**
   - After satisfactory metrics (>95% string-level accuracy):
     ```python
     model.export(format="onnx", opset=12, imgsz=640, dynamic=False, simplify=False, name="models/best-ocr.onnx")
     model.export(format="tflite", imgsz=640, int8=True, name="models/best-ocr-int8.tflite")
     ```

7. **React Native integration update**
   - Add next to the current detection-+-MLKit pipeline a **single** `detectCharacters()` call that outputs `[detected_char, confidence, box]` for all 11 detections.
   - Simplify post-processing: sort by `y_center`, concatenate the detected characters, compare with filename.
   - Keep `stitchCharacters()` and MLKit OCR modules and separate the pipelines (keep feature-flagged for A/B testing common).

8. **Timeline & Deliverables**
   | Phase | Owner | Duration |
   |-------|-------|----------|
   | Data conversion script | TS/Python | 0.5 day |
   | Notebook duplication + training | Marcel | 2–3 days |
   | Validation script | TS/Python | 0.5 day |
   | Export & mobile integration | RN dev | 1 day |
   | Accuracy / perf validation | QA | 1 day |

9. **Success criteria**
   1. ≥97 % string-level accuracy on the 500-image validation set.
   2. Average inference time ≤400 ms on target Android device.
   3. Model size ≤15 MB (TFLite int8).

> After this upgrade the entire on-device pipeline will be pure **YOLO-OCR** with zero external OCR dependencies, drastically reducing runtime, memory, and network usage. 