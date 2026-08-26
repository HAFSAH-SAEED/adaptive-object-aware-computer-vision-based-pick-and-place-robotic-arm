# Adaptive Object-Aware Vision-Guided Pick-and-Place Robot

## Project Overview

This project focuses on developing an adaptive, vision-guided robotic pick-and-place system capable of identifying objects, estimating their visual properties and position, and eventually translating this information into robot motion commands.

The planned system combines computer vision, object recognition, geometric reasoning, coordinate mapping, inverse kinematics, and robotic control.

## Current Development Status

The initial software-based computer vision implementation has been developed and tested successfully.

The current implementation includes:

- Image acquisition using OpenCV
- BGR to HSV color-space conversion
- Color-based object segmentation
- Morphological noise removal
- Contour detection
- Contour area filtering
- Bounding-box extraction
- Centroid calculation using image moments
- Basic shape classification
- Pixel-based object size estimation
- Real-time object detection using a live camera feed

The current computer vision pipeline has been tested using a blue circular object and has also been demonstrated using a live laptop camera.

An iPhone camera has been successfully connected to the PC through DroidCam. Direct integration of the iPhone feed into the Python/OpenCV pipeline is currently in progress.

## Current Vision Pipeline

Image / Camera Frame
↓
BGR → HSV
↓
Color Thresholding
↓
Morphological Processing
↓
Contour Detection
↓
Area Filtering
↓
Bounding Box
↓
Centroid Extraction
↓
Shape Classification
↓
Pixel-Based Size Estimation

## Planned System

Camera
↓
Image Acquisition
↓
Preprocessing
↓
Object Detection / Recognition
↓
Color + Shape + Size Estimation
↓
Centroid Extraction
↓
Camera Calibration
↓
Homography / Coordinate Mapping
↓
Error Mapping
↓
Robot Coordinates
↓
Inverse Kinematics
↓
Robot Simulation
↓
Pick-and-Place Control

## Object Information

The intended object representation will include:

- Object type
- Color
- Shape
- Size
- Image-space position
- Confidence
- Mapped workspace position

## Hardware Implementation

The current stage focuses on establishing and validating the software and computer-vision pipeline.

The initial hardware implementation is planned to begin in September 2026. This stage will involve integrating the vision system with the robotic platform, calibration setup, robot simulation/control, and eventually the physical pick-and-place system.

## Development Approach

The project is being developed incrementally, beginning with classical computer-vision techniques and progressively integrating more advanced object recognition, camera calibration, coordinate transformation, and robotic control components.

## Technologies

- Python
- OpenCV
- NumPy
- Computer Vision
- Camera Calibration
- Homography
- Object Detection
- Robotics
- Inverse Kinematics
- Arduino
- Raspberry Pi
- Robot Simulation

## Project Status

Current phase: Initial computer-vision software implementation

Camera integration: In progress

Hardware implementation: Planned for September 2026

Overall project: In development