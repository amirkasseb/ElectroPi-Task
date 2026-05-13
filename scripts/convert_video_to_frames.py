import cv2
import os

def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

video_path = "video/Task.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frames = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    score = sharpness(frame)

    frames.append((score, frame_id, frame))

    frame_id += 1

cap.release()

frames.sort(reverse=True, key=lambda x: x[0])

top_30 = frames[:30]

for i, (_, _, frame) in enumerate(top_30):
    cv2.imwrite(f"{output_folder}/frame_{i:03d}.jpg", frame)

print("Saved 30 best sharp frames")