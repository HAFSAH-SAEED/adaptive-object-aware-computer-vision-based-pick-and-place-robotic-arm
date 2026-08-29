# Adaptive Object-Aware Vision-Guided Pick-and-Place Robotic Arm

A vision-guided robotic pick-and-place system using an overhead camera, YOLO-based object detection, classical computer vision, camera calibration, homography-based coordinate mapping, and color matching.

The current implementation focuses on object detection, localization, real-world coordinate estimation, and an initial object-to-target matching demonstration. Inverse kinematics and robotic hardware integration are the next development stages.

## Current Pipeline

Camera → Image Acquisition → Image Preprocessing → YOLO Object Detection → Color + Shape Recognition → Centroid Extraction → Camera Calibration → Homography / Coordinate Mapping → Real-World X,Y Coordinates → Target Detection → Color Matching → Pick & Place Coordinates

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

A custom YOLO11n model was trained to detect three object classes:

- RED
- GREEN
- BLUE

Dataset:

- 43 images
- 34 training images
- 9 validation images
- YOLO bounding-box labels

The trained model is stored in `models/best.pt`.

### YOLO Validation Results

- Precision: 0.988
- Recall: 1.000
- mAP50: 0.995
- mAP50-95: 0.907

These results are based on the current small validation dataset.

## Object Localization

For each detected object, the system extracts:

- Color
- Confidence
- Shape
- Pixel centroid
- Real-world X coordinate
- Real-world Y coordinate

The detected pixel centroid is transformed using the calibrated homography to obtain real-world coordinates in millimeters.

Example:

- RED → X: 62.2 mm, Y: 187.7 mm
- GREEN → X: 191.6 mm, Y: 185.8 mm
- BLUE → X: 56.0 mm, Y: 100.0 mm

## Target Detection & Matching

A separate classical computer vision stage has been added for target detection.

The current demonstration uses one BLUE target. The target is detected separately from the YOLO objects using color-based image processing and contour detection.

The target position is then converted into the same real-world coordinate system using the existing homography.

### Current Matching Demonstration

**BLUE OBJECT → BLUE TARGET**

Example:

**Blue Object**

Pick position: X = 56.0 mm, Y = 100.0 mm

**Blue Target**

Place position: X = 183.8 mm, Y = 90.8 mm

The RED and GREEN objects continue to be detected, but they currently have no matching targets in the demonstration setup.

## Current System

The current implementation can:

1. Detect RED, GREEN, and BLUE objects.
2. Identify object color and shape.
3. Extract object centroids.
4. Convert object coordinates from pixels to real-world X,Y coordinates.
5. Detect a BLUE target using classical computer vision.
6. Extract the target centroid.
7. Convert the target position into real-world coordinates.
8. Match the BLUE object to the BLUE target.
9. Generate pick and placement coordinates.

## Project Structure

The project contains the YOLO model, dataset, calibration, detection, coordinate-mapping, and matching components.

Main files:

- `calibrate.py` — camera/workspace calibration and homography generation
- `camera_test.py` — camera testing
- `collect_yolo_data.py` — custom YOLO dataset collection
- `coordinate_test.py` — coordinate testing
- `find_camera.py` — camera identification/testing
- `live_detection.py` — live detection
- `main.py` — main project pipeline
- `multi_color_detection.py` — multi-color classical CV detection
- `split_dataset.py` — YOLO dataset splitting
- `test_yolo.py` — integrated YOLO, localization, target detection, and matching pipeline
- `data.yaml` — YOLO dataset configuration
- `models/best.pt` — trained YOLO11n model
- `yolo_dataset/` — training and validation dataset

## Running the Current System

Activate the virtual environment and run:

`python test_yolo.py`

The program displays:

- Calibrated workspace boundary
- Detected objects
- Confidence scores
- Object shapes
- Object centroids
- Real-world X,Y coordinates
- BLUE target
- Target coordinates
- Matching information

The terminal reports the pick and placement coordinates for the current BLUE object → BLUE target demonstration.

## Calibration

If the camera or workspace position changes, recalibrate the system using:

`python calibrate.py`

The updated calibration data is automatically loaded by the detection pipeline.

The local calibration file `calibration_data.npz` is excluded from GitHub because it depends on the specific camera and workspace configuration.

## Next Steps

The next development stages are:

Current Vision System → Inverse Kinematics → Robot Simulation / Validation → Raspberry Pi + Linux → Raspberry Pi ↔ Arduino Serial Communication → Servo Motor Control → 6-DOF Robotic Arm → Physical Pick-and-Place

Future improvements include:

- Multiple target detection
- Dynamically shuffled target locations
- Multiple object-to-target mappings
- Improved target detection robustness
- Inverse kinematics
- Robot simulation
- Collision and workspace constraints
- Raspberry Pi deployment
- Arduino servo control
- Complete autonomous pick-and-place

## Status

**Vision + Localization + Initial Matching: Implemented**

**Inverse Kinematics: Next Stage**

**Hardware Pick-and-Place: Planned**

## Author

**Designed and developed by Hafsah Saeed, Computer Engineering student at NUST.**

*Turning computer vision into real-world robotic intelligence.*