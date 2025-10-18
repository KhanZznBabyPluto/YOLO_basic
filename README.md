# YOLOv8 Multi-Object Tracking with Kalman Filter and Custom Metric

This repository contains a multi-object tracking solution using YOLOv8 on the Waymo Open Dataset, enhanced with a Kalman Filter for trajectory smoothing and a custom metric combining IoU and Jitter. The project compares the proposed method with a baseline approach.

## Overview
- **Dataset**: Waymo Open Dataset (training segment, ~1000 frames).
- **Model**: Pretrained YOLOv8n (no fine-tuning yet).
- **Enhancements**: Kalman Filter for tracking stability, custom metric (0.6 * IoU + 0.3 * (1 - Jitter) + 0.1 * ContextScore).
- **Baseline**: Standard YOLOv8 tracking without Kalman.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KhanZznBabyPluto/YOLO_basic.git
   cd YOLO_basic
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the Waymo Open Dataset TFRecord files and update `tfrecord_paths` in `main.py`.

## Usage
Run the main script to process data and generate results:
```bash
python3 main.py
```

## Results
- **Videos**:
  - [Kalman Tracking](https://youtu.be/r3TyD34WdHU) (upload `output/track_test_kalman.mp4` to YouTube).
  - [Baseline Tracking](https://youtu.be/kJYEnY9YoNg) (upload `output/track_test_baseline.mp4` to YouTube).
- **Metrics Graph**: [metrics.png](output/metrics.png) shows Custom Metric, IoU, and 1 - Jitter over time for both methods.

### Performance Comparison
| Method                | IoU    | Jitter | Custom Metric |
|-----------------------|--------|--------|---------------|
| Baseline (YOLOv8)     | 0.02   | 0.40   | 0.25          |
| Kalman + Custom Metric| 0.15   | 0.30   | 0.35          |

- **Notes**: Values are approximate based on 1000 frames. Kalman improves stability (lower Jitter) and overall metric.

## Files
- `main.py`: Main script to run tracking and visualization.
- `track_yolo.py`: Implements YOLOv8 tracking with and without Kalman Filter.
- `metrics.py`: Computes custom metric (IoU + 1-Jitter + ContextScore).
- `visualize.py`: Generates video and metrics plot.
- `waymo_parsing.py`: Parses Waymo TFRecord files.

## Future Work
- **Fine-tuning**: Train YOLOv8 on Waymo data to improve IoU.
- **More Data**: Expand to multiple TFRecord files for robustness.
- **Context Integration**: Enhance metric with weather/lighting data.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Waymo Open Dataset: https://waymo.com/open/
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- FilterPy for Kalman Filter: https://github.com/rlabbe/filterpy

## Contact
For questions or contributions, open an issue or contact [khangeldinansar@gmail.com](mailto:khangeldinansar@gmail.com).
```