import torch
from torchvision.ops import box_iou

def compute_metrics(gt_bboxes, pred_bboxes, track_ids, prev_pred_bboxes=None, prev_track_ids=None, frame_context=None):
    """
    Calculates custom metric: 0.6*IoU + 0.3*(1-Jitter) + 0.1*ContextScore.
    Args:
        gt_bboxes (list[dict]): List of ground truth bounding boxes for current frame.
        pred_bboxes (list[dict]): List of predicted bounding boxes for current frame.
        track_ids (list[int]): List of track IDs corresponding to predicted bboxes in current frame.
        prev_pred_bboxes (list[dict], optional): Predicted bboxes from previous frame.
        prev_track_ids (list[int], optional): Track IDs from previous frame.
        frame_context (object, optional): Frame context object with weather/lighting info.
    Returns:
        tuple: (metric, iou, one_minus_jitter)
    """
    if not gt_bboxes or not pred_bboxes:
        return 0.0, 0.0, 0.0
    
    # IoU
    gt_boxes = torch.tensor([[b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']] for b in gt_bboxes])
    pred_boxes = torch.tensor([[b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']] for b in pred_bboxes])
    iou = box_iou(gt_boxes, pred_boxes).mean().item() if len(gt_boxes) > 0 and len(pred_boxes) > 0 else 0.0
    
    # Jitter
    one_minus_jitter = 0.0
    if prev_pred_bboxes and prev_track_ids:
        curr_centers = {b['track_id']: (b['x'] + b['w']/2, b['y'] + b['h']/2) for b in pred_bboxes}
        prev_centers = {b['track_id']: (b['x'] + b['w']/2, b['y'] + b['h']/2) for b in prev_pred_bboxes}
        common_ids = set(track_ids) & set(prev_track_ids)
        
        if common_ids:
            jitter_sum = 0.0
            for tid in common_ids:
                if tid in curr_centers and tid in prev_centers:
                    dx = curr_centers[tid][0] - prev_centers[tid][0]
                    dy = curr_centers[tid][1] - prev_centers[tid][1]
                    dist = (dx**2 + dy**2)**0.5
                    
                    curr_bbox = next(b for b in pred_bboxes if b['track_id'] == tid)
                    norm_factor = max(curr_bbox['w'], curr_bbox['h']) if max(curr_bbox['w'], curr_bbox['h']) > 0 else 1.0
                    norm_dist = min(dist / norm_factor, 1.0)
                    jitter_sum += norm_dist
            
            avg_jitter = jitter_sum / len(common_ids)
            one_minus_jitter = 1.0 - avg_jitter
        else:
            one_minus_jitter = 1.0
    else:
        one_minus_jitter = 1.0
    
    # Context score
    context_score = 1.0
    if frame_context:
        weather = frame_context.weather
        if weather.rain_intensity > 0 or weather.is_night:
            context_score = 0.5
    
    # My custom metric
    metric = 0.6 * iou + 0.3 * one_minus_jitter + 0.1 * context_score
    
    return metric, iou, one_minus_jitter