import cv2
import matplotlib.pyplot as plt

def save_video(images, yolo_bboxes, track_ids, output_path):
    """
    Сохраняет видео с bbox и track IDs.
    Args:
        images (list): Список np.array (RGB изображения).
        yolo_bboxes (list): Список bbox [{frame_idx, x, y, w, h, class, track_id}, ...].
        track_ids (list): Список track IDs для каждого кадра.
        output_path (str): Путь для сохранения видео.
    """
    h, w = images[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), 10, (w, h))

    for img, bboxes, ids in zip(images, yolo_bboxes, track_ids):
        img_copy = img.copy()
        for box, tid in zip(bboxes, ids):
            x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])

            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_copy, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    out.release()

def plot_metrics(metrics, iou_list, jitter_list, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics, label='Custom Metric (w1 *IoU + w2 * (1-Jitter))', color='blue')
    plt.plot(iou_list, label='IoU Component', color='green', linestyle='--')
    plt.plot(jitter_list, label='1 - Jitter Component', color='red', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Tracking Quality Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()

def plot_several_metrics(metrics_list, iou_list, jitter_list, output_path):
    plt.figure(figsize=(12, 6))
    for i, (metrics, iou, jitter) in enumerate(zip(metrics_list, iou_list, jitter_list)):
        label = 'Kalman + Custom Metric' if i == 0 else 'Baseline'
        plt.plot(metrics, label=f'{label} (Custom Metric)', color=['blue', 'orange'][i])
        plt.plot(iou, label=f'{label} (IoU)', color=['green', 'cyan'][i], linestyle='--')
        plt.plot([1 - j for j in jitter], label=f'{label} (1 - Jitter)', color=['red', 'magenta'][i], linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Tracking Quality Over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()