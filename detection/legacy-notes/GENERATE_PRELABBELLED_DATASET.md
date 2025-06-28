# Generate a pre-labelled dataset from our existing models to scan vertical ocean container codes

We use Roboflow for labelling and want to really get a large dataset in there now (>2000 image). Pre-labelling speeds up labelling tremendously. We have a model that identifies the (most likely) area where a character may be. And our filenames contain the actual information of the container.

Input parameters:
* folder with images: data/dataset/images/raw
* file with target classes: models/classes.txt
* YOLO model that identifies the areas: models/best-OCR-14-06-25.pt

Output:
* We want YOLO8 labels for each of these images in the label output folder: data/dataset/labels/raw

We need a python script to loop through our RAW folder and for each image:

* copy image to data/dataset/images/to-train
* run our YOLO model against it: this will identify the top-areas where characters may be.
* filter the top 11 candidates, for those 'vertically alligned' 
* use the filename to label each to re-label the 11 boxes top-down (first character of the filename corresponds to the top box) from their "character" to the actual character. Use models/classes.txt as the label index
* place the label in data/dataset/labels/to-train
