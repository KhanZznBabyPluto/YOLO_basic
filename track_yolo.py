import numpy as np

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

def run_kalman_tracking(images, model_path='yolov8n.pt'):
    """
    Выполняет трекинг объектов с YOLOv8.
    Args:
        images (list): Список np.array (RGB изображения).
        model_path (str): Путь к YOLO модели.
    Returns:
        yolo_bboxes (list): Список bbox [{frame_idx, x, y, w, h, class, track_id}, ...].
        track_ids (list): Список track IDs для каждого кадра.
    """
    model = YOLO(model_path)
    yolo_bboxes, track_ids = [], []
    trackers = {}

    for frame_idx, img in enumerate(images):
        results = model.track(img, persist=True)
        frame_bboxes = []
        frame_ids = []

        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls = int(box.cls.item())
            track_id = int(box.id.item()) if box.id is not None else -1

            if track_id not in trackers:
                kf = KalmanFilter(dim_x=4, dim_z=2)
                kf.x = np.array([x, y, 0, 0])
                kf.F = np.array([[1, 0, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
                
                kf.H = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]])

                kf.P *= 1000.
                kf.R = np.array([[10., 0],
                                 [0, 10.]])
                trackers[track_id] = kf
                print(f"Initialized Kalman for track_id {track_id} at (x={x}, y={y})")
            else:
                kf = trackers[track_id]
                kf.predict()
                kf.update(np.array([x, y]))
                x, y = kf.x[:2]

            frame_bboxes.append({
                'frame_idx': frame_idx,
                'x': x - w / 2,
                'y': y - h / 2,
                'w': w,
                'h': h,
                'class': cls,
                'track_id': track_id
            })
            frame_ids.append(track_id)
        
        yolo_bboxes.append(frame_bboxes)
        track_ids.append(frame_ids)
    
    return yolo_bboxes, track_ids

def run_baseline(images, model_path='yolov8n.pt'):
    """
    Выполняет трекинг объектов с YOLOv8.
    Args:
        images (list): Список np.array (RGB изображения).
        model_path (str): Путь к YOLO модели.
    Returns:
        yolo_bboxes (list): Список bbox [{frame_idx, x, y, w, h, class, track_id}, ...].
        track_ids (list): Список track IDs для каждого кадра.
    """
    model = YOLO(model_path)
    yolo_bboxes, track_ids = [], []

    for frame_idx, img in enumerate(images):
        results = model.track(img, persist=True)
        frame_bboxes = []
        frame_ids = []

        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls = int(box.cls.item())
            track_id = int(box.id.item()) if box.id is not None else -1

            frame_bboxes.append({
                'frame_idx': frame_idx,
                'x': x - w / 2,
                'y': y - h / 2,
                'w': w,
                'h': h,
                'class': cls,
                'track_id': track_id
            })
            frame_ids.append(track_id)
        
        yolo_bboxes.append(frame_bboxes)
        track_ids.append(frame_ids)
    
    return yolo_bboxes, track_ids