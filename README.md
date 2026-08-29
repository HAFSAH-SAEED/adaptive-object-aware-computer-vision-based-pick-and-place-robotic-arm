# Adaptive Object-Aware Vision-Guided Pick-and-Place Robotic Arm

A vision-guided robotic pick-and-place system using an overhead camera, YOLO-based object detection, classical computer vision, camera calibration, homography-based coordinate mapping, and color matching.

The current implementation focuses on object detection, localization, real-world coordinate estimation, and an initial object-to-target matching demonstration. Robotic control and inverse kinematics are planned next.

---

## Current Pipeline

Camera
↓
Image Acquisition
↓
Image Preprocessing
↓
YOLO Object Detection
↓
Color + Shape Recognition
↓
Centroid Extraction
↓
Camera Calibration
↓
Homography / Coordinate Mapping
↓
Real-World X,Y Coordinates
↓
Target Detection
↓
Color Matching
↓
Pick & Place Coordinates

---

## Implemented

### Computer Vision

- Camera/frame acquisition
- Image preprocessing
- BGR → HSV conversion
- Color thresholding
- Morphological processing
- Contour detection
- Area filtering
- Bounding-box extraction
- Centroid extraction
- Shape classification
- Pixel-based size estimation
- Camera/workspace calibration
- Homography-based perspective mapping
- Pixel → real-world X,Y conversion

### YOLO Object Detection

A custom YOLO11n model was trained to detect:

- RED
- GREEN
- BLUE

Dataset:

- 43 images
- 34 training images
- 9 validation images
- YOLO bounding-box labels

The trained model is stored at:

`models/best.pt`

Training results:

- Precision: 0.988
- Recall: 1.000
- mAP50: 0.995
- mAP50-95: 0.907

### Target Detection & Matching

A separate classical-CV target detection stage has been implemented.

Current demonstration:

**BLUE OBJECT → BLUE TARGET**

The system calculates both positions in the same real-world coordinate system.

Example:

```text
BLUE OBJECT
Pick:  X = 56.0 mm
       Y = 100.0 mm

BLUE TARGET
Place: X = 183.8 mm
       Y = 90.8 mm

## Author

**Designed and developed by Hafsah Saeed, Computer Engineering student at NUST.**  
*Turning computer vision into real-world robotic intelligence.*