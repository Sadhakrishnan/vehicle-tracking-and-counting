import cv2
import numpy as np
from ultralytics import YOLO
import datetime

# 1️⃣ Load YOLOv11 model (make sure you have 'yolo11n.pt' downloaded)
model = YOLO("yolo11x.pt")

# 2️⃣ Define vehicle classes and mapping from COCO indices to names
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# 3️⃣ Virtual counting line (y-coordinate)
LINE_Y = 600

# 4️⃣ Initialize counters
class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
total_count = 0

# 5️⃣ Keep track of which IDs have crossed
counted_ids = set()
last_positions = {}

# 6️⃣ Begin ByteTrack via YOLOv11.track
results = model.track(
    source="1.mp4",             # Input video
    tracker="bytetrack.yaml",   # Use ByteTrack
    classes=list(VEHICLE_CLASSES.keys()),
    conf=0.5,
    verbose=False,
    stream=True,                # Stream results frame-by-frame
    show=False                  # We'll display via OpenCV manually
)
writer = None

# 7️⃣ Process each frame from the tracker
for result in results:
    frame = result.orig_img
    if frame is None:
        break

    # Initialize writer once we get frame dimensions
    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30, (w, h))

    start = datetime.datetime.now()

    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0,255,255), 2)
    if result.boxes.id is None:
        continue
    # ✅ Process tracked objects in this frame
    ids = result.boxes.id.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    bboxes = result.boxes.xyxy.cpu().numpy()

    for tid, cls_id, box in zip(ids, classes, bboxes):
        cls_id = int(cls_id)
        track_id = int(tid)
        if cls_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[cls_id]
        x1, y1, x2, y2 = map(int, box)
        center_y = (y1 + y2) // 2
        prev_y = last_positions.get(track_id, center_y)
        last_positions[track_id] = center_y

        # Count when crossing downward
        if prev_y < LINE_Y <= center_y and track_id not in counted_ids:
            counted_ids.add(track_id)
            class_counts[label] += 1
            total_count += 1

        # Draw box & label
        color = tuple(int(c) for c in np.random.RandomState(42 + cls_id).randint(0,255,3))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{track_id}-{label}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display counts in top-left
    ypos = 30
    for v, cnt in class_counts.items():
        cv2.putText(frame, f"{v.capitalize()}: {cnt}", (10, ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        ypos += 30
    cv2.putText(frame, f"Total: {total_count}", (10, ypos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Display FPS
    elapsed = (datetime.datetime.now() - start).total_seconds()
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1]-150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Show and save
    cv2.imshow("Vehicle Counter", frame)
    writer.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
writer.release()
