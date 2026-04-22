# EMTracker: Enhanced Multi-Object Tracker with Soft Attention & Interpolation

**EMTracker** is a robust, hybrid Multi-Object Tracking (MOT) system designed for complex, dynamic environments with severe occlusions (e.g., racing, rally, heavy dust, or crowded scenes). 

It combines YOLO-based detections, a custom Soft-Attention Re-Identification (Re-ID) module, and bidirectional interpolation to maintain track IDs even when objects disappear for extended periods.

## ✨ Key Features

* **Soft-Attention Re-ID:** Utilizes `MobileNetV3-Small` paired with a generated Gaussian mask to focus embedding extraction on the center of the bounding box, reducing background noise.
* **Spatio-Temporal Association:** Matches tracks using a unified cost matrix calculating visual similarity, spatial distance, and IoU.
* **Bidirectional Track Interpolation:** A post-processing engine that mathematically bridges gaps in tracks (up to 50 frames by default) using linear interpolation, drastically reducing ID Switches during full occlusions.
* **Custom Metrics Support:** Native integration with `motmetrics` and a custom implementation for calculating **HOTA** (Higher Order Tracking Accuracy).

## 🛠️ Architecture

1. **Detection:** YOLOv8 (custom weights supported).
2. **Feature Extraction:** Cropped detections are masked with a Gaussian kernel and passed through MobileNetV3 to generate high-quality embeddings.
3. **Tracking (RobustMasterTracker):** Hungarian algorithm assigns detections to existing tracks based on similarity thresholds and distance penalties. Unmatched detections spawn new tracks.
4. **Post-Processing (InterpolativeTracker):** The history of all tracks is analyzed, and missing spatial coordinates between known states are mathematically filled.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/EMTracker.git](https://github.com/yourusername/EMTracker.git)
   cd EMTracker
