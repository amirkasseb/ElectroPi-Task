from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import time
from collections import deque, Counter

# ======================
# CONFIG
# ======================
model = YOLO("model/best.pt")

CONF_THRESH = 0.2
IOU_THRESH = 0.15

video_path = "/content/drive/MyDrive/Task.mp4"
output_path = "Output/output.mp4"

os.makedirs("Output", exist_ok=True)

# ======================
# OPEN VIDEO
# ======================
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = None

# ======================
# OUTPUT VIDEO
# ======================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ======================
# MODE FILTER MEMORY
# ======================
history = deque(maxlen=20)

# ======================
# PROGRESS
# ======================
pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

# ======================
# PROCESS VIDEO
# ======================
frame_id = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ======================
    # YOLO PREDICTION
    # ======================
    results = model.predict(
        frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False
    )

    r = results[0]
    annotated = r.plot()

    # ======================
    # RAW COUNT
    # ======================
    raw_count = len(r.boxes) if r.boxes is not None else 0

    # ======================
    # MODE FILTER (STABLE COUNT)
    # ======================
    history.append(raw_count)
    stable_count = Counter(history).most_common(1)[0][0]

    # ======================
    # FPS
    # ======================
    elapsed = time.time() - start_time
    fps_now = frame_id / elapsed if elapsed > 0 else 0

    # ======================
    # UI DRAW
    # ======================
    cv2.rectangle(annotated, (10, 10), (260, 90), (0, 0, 0), -1)

    cv2.putText(
        annotated,
        f"Count: {stable_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps_now:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ======================
    # SAVE
    # ======================
    out.write(annotated)

    pbar.update(1)
    print(f"Frame: {frame_id}", end="\r")

# ======================
# CLEANUP
# ======================
cap.release()
out.release()
pbar.close()

print("\n✅ Done")
print(f"📁 Saved: {output_path}")