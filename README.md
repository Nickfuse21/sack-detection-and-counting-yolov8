# Sack Detection and Counting System using YOLOv8

## Project Overview

This project implements a computer vision-based system for detecting and counting sacks from industrial video footage. The solution uses the YOLOv8 object detection model combined with custom filtering and centroid-based tracking logic to improve sack identification and prevent duplicate counting.

The primary objective of this implementation is to build a functional and stable prototype capable of:

- Detecting sack-like objects in video frames
- Ignoring humans and large vehicles
- Preventing duplicate counting across frames
- Displaying real-time annotated detection output
- Saving processed video results

This project focuses on practical implementation, real-time performance, and structured engineering workflow suitable for industrial scenarios.

---

## Problem Statement

In industrial environments, monitoring and counting sacks manually can be inefficient and error-prone. The goal of this project is to automate sack detection and counting using computer vision techniques applied to recorded video streams.

The system processes video frames sequentially, detects potential sack-like objects, applies intelligent filtering, and maintains a running count of unique sacks.

---

## Methodology

### 1. Object Detection

The system uses the YOLOv8 nano model pretrained on the COCO dataset. Although the COCO dataset does not include a dedicated "sack" class, sack-like objects are identified using detection confidence combined with additional filtering logic.

### 2. Intelligent Filtering

Since the pretrained model detects multiple object categories (such as persons, vehicles, etc.), additional filtering steps are implemented:

- Human detections are explicitly ignored
- Confidence threshold removes low-probability detections
- Area-based filtering eliminates very small and very large objects
- Aspect ratio filtering removes elongated shapes such as trucks or vehicles

These filters help narrow detections to sack-like objects while maintaining reasonable recall.

### 3. Centroid-Based Tracking

To prevent duplicate counting:

- The centroid of each detected object is calculated
- A new object is counted only if its centroid does not closely match previously stored centroids
- A distance threshold ensures the same sack is not counted multiple times across frames

This lightweight tracking approach provides stable counting without requiring advanced tracking algorithms.

---

## System Workflow

1. Load pretrained YOLOv8 model
2. Read video frame-by-frame
3. Perform object detection on each frame
4. Apply class, confidence, area, and aspect ratio filtering
5. Compute centroid and compare with previously tracked objects
6. Update total sack count
7. Display live annotated frame
8. Save processed output video

---

## Project Structure

Bag_Counting_Project/
│
├── sack_count_api.py
├── requirements.txt
├── README.md
└── video/
    └── Problem Statement Scenario3.mp4

---

## Installation

It is recommended to use a virtual environment.

Create and activate virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
