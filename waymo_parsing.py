import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

def get_paths(dir_name, num_files=None):
    paths = []

    for item in os.listdir(dir_name):
        full_path = os.path.join(dir_name, item)
        paths.append(full_path)

        if num_files is not None and len(paths) >= num_files:
            break

    return paths

def parse_tfrecord(tfrecord_paths, max_frames=100, class_filter=None):
    """
    Parse several TFRecord Waymo videos, extracte images (FRONT camera) and 2D bbox.
    Args:
        tfrecord_paths (list or str): Paths to TFRecord files.
        max_frames (int): Maximum of frames to be processed per video.
        class_filter (list): List of classes for filtering.
    Returns:
        images (list): List np.array (RGB images).
        gt_bboxes (list): List bbox [{frame_idx, x, y, w, h, class}, ...].
    """

    if isinstance(tfrecord_paths, str):
        tfrecord_paths = [tfrecord_paths]

    images, gt_bboxes = [], []
    global_frame_idx = 0

    for tfrecord_path in tfrecord_paths:
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

        for frame_idx, data in enumerate(dataset):
            if frame_idx >= max_frames:
                break
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            found_image = False
            for img in frame.images:
                if img.name == dataset_pb2.CameraName.FRONT:
                    img_np = tf.image.decode_jpeg(img.image).numpy()
                    # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    images.append(img_np)
                    found_image = True
                    break
            if not found_image:
                continue

            frame_bboxes = []
            for label in frame.camera_labels:
                if label.name == dataset_pb2.CameraName.FRONT:
                    for lb in label.labels:
                        if class_filter is None or lb.type in class_filter:
                            frame_bboxes.append({
                                'frame_idx': global_frame_idx,
                                'x': lb.box.center_x,
                                'y': lb.box.center_y,
                                'w': lb.box.length,
                                'h': lb.box.width,
                                'class': lb.type,
                            })
            gt_bboxes.append(frame_bboxes)
            global_frame_idx += 1

    return images, gt_bboxes

if __name__ == "__main__":
    tfrecord_path = "/home/ansar/vscode_projects/pet_projects/Challenge/Waymo/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
    images, gt_bboxes = parse_tfrecord(tfrecord_path, max_frames=50, class_filter=[1, 2])
    print(f"Извлечено {len(images)} кадров, {sum(len(b) for b in gt_bboxes)} bbox")
                
                
                