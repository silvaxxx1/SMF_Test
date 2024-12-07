import cv2
import os

# Path to your video
video_path = "orj_1.mp4"
output_folder = "extracted_frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Capture video
cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save every nth frame (e.g., every 5th frame for 30 FPS video)
    if frame_count % 5 == 0:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print(f"Frames saved to {output_folder}")
