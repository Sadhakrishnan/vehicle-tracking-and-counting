# inference.py
import numpy as np
import datetime
import cv2
from ultralytics import YOLO
from collections import defaultdict

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

from helper import create_video_writer

VEHICLE_CLASSES = {"car", "motorbike", "bus", "truck"}
conf_threshold = 0.5
LINE_Y = 450

# Load COCO class names
with open("config/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Pre-load colors
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))

def process_video(video_path: str, output_path: str = "output.mp4"):
    # Tracker & Models
    model = YOLO("yolov8s.pt")
    encoder = gdet.create_box_encoder("config/mars-small128.pb", batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker = Tracker(metric)

    video_cap = cv2.VideoCapture(video_path)
    writer = create_video_writer(video_cap, output_path)

    vehicle_counter = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
    total_counter = 0
    counted_ids = set()
    last_y_positions = {}

    while True:
        ret, frame = video_cap.read()
        if not ret or frame is None:
            print("End of video.")
            break

        start_time = datetime.datetime.now()
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)

        results = model(frame)
        for result in results:
            bboxes, confidences, class_ids = [], [], []
            for data in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = data
                class_name = class_names[int(class_id)]
                if confidence > conf_threshold and class_name in VEHICLE_CLASSES:
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    bboxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(int(class_id))

        names = [class_names[i] for i in class_ids]
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, conf, name, feat) for bbox, conf, name, feat in zip(bboxes, confidences, names, features)]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name not in VEHICLE_CLASSES:
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, bbox)
            center_y = int((y1 + y2) / 2)

            prev_y = last_y_positions.get(track_id, center_y)
            last_y_positions[track_id] = center_y

            if prev_y < LINE_Y <= center_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_counter[class_name] += 1
                total_counter += 1

            color = colors[class_names.index(class_name)]
            B, G, R = map(int, color)
            label = f"{track_id}-{class_name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (int(B), int(G), int(R)), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (int(B), int(G), int(R)), 2)

        # Counter overlay
        y_pos = 30
        for k, v in vehicle_counter.items():
            cv2.putText(frame, f"{k.capitalize()}: {v}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        cv2.putText(frame, f"Total: {total_counter}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        fps = 1 / (datetime.datetime.now() - start_time).total_seconds()
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)

    video_cap.release()
    writer.release()
    return output_path

