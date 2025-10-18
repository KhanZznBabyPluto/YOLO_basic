from waymo_parsing import get_paths, parse_tfrecord
from track_yolo import run_kalman_tracking, run_baseline
from visualize import plot_metrics, save_video, plot_several_metrics
from metrics import compute_metrics

def main():
    tfrecord_paths = get_paths("training")
    max_frames_per_file = 200

    images, gt_bboxes = parse_tfrecord(tfrecord_paths, max_frames_per_file, class_filter=[1, 2])

    yolo_bboxes, track_ids = run_kalman_tracking(images, model_path='yolov8n.pt')

    metrics, iou_list, jitter_list = [], [], []
    prev_pred_bboxes, prev_track_ids = None, None
    for frame_idx, (gt, pred, ids) in enumerate(zip(gt_bboxes, yolo_bboxes, track_ids)):
        metric, iou, jitter = compute_metrics(gt, pred, ids, prev_pred_bboxes, prev_track_ids)
        metrics.append(metric)
        iou_list.append(iou)
        jitter_list.append(jitter)

        prev_pred_bboxes = pred
        prev_track_ids = ids
    
    yolo_bboxes_base, track_ids_base = run_baseline(images, model_path='yolov8n.pt')

    metrics_base, iou_list_base, jitter_list_base = [], [], []
    prev_pred_bboxes_base, prev_track_ids_base = None, None
    for frame_idx, (gt, pred, ids) in enumerate(zip(gt_bboxes, yolo_bboxes_base, track_ids_base)):
        metric, iou, jitter = compute_metrics(gt, pred, ids, prev_pred_bboxes_base, prev_track_ids_base)
        metrics_base.append(metric)
        iou_list_base.append(iou)
        jitter_list_base.append(jitter)

        prev_pred_bboxes_base = pred
        prev_track_ids_base = ids
    
    save_video(images, yolo_bboxes, track_ids, 'output/track_test_kalman.mp4')
    save_video(images, yolo_bboxes_base, track_ids_base, 'output/track_test_baseline.mp4')
    
    plot_several_metrics(
        [metrics, metrics_base],
        [iou_list, iou_list_base],
        [jitter_list, jitter_list_base],
        'output/metrics.png'
    )

if __name__ == "__main__":
    main()
