import cv2
import os
import tempfile
from roboflow import Roboflow
from PIL import Image

# Initialize Roboflow client
rf = Roboflow(api_key="VN28ceooZnimGyMKzkNo")
project = rf.workspace().project("smart-meta-factory")
model = project.version(2).model

# Open video file
video_input = "orj_1.mp4"
cap = cv2.VideoCapture(video_input)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
output_video = "output_video_final2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Target classes
target_classes = ['hairnet', 'helmet', 'mask_correct', 'mask_incorrect', 'safety vest']

# Create a single temporary file to reuse
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save the current frame to the temporary file
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    pil_image.save(temp_file.name)

    # Perform inference using Roboflow API
    predictions = model.predict(temp_file.name, confidence=50, overlap=45).json()

    # Initialize detection flags and person box
    current_frame_labels = {label: False for label in target_classes}
    person_box = None

    if 'predictions' in predictions:
        for prediction in predictions['predictions']:
            x, y, w, h = (int(prediction[k]) for k in ['x', 'y', 'width', 'height'])
            label = prediction['class']
            confidence = prediction['confidence']

            # Handle the 'person' bounding box
            if label == "person" and confidence > 0.6:
                person_box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)

            # Update detected labels
            if label in target_classes and confidence > 0.6:
                current_frame_labels[label] = True

    # Draw the 'person' bounding box and display labels above it
    if person_box is not None:
        cv2.rectangle(frame, (person_box[0], person_box[1]),
                      (person_box[2], person_box[3]), (0, 255, 0), 2)

        # Display labels above the bounding box
        y_offset = person_box[1] - 10
        label_order = ["Accessories", "mask", "safety vest", "helmet", "hairnet"]

        for label in label_order:
            if label == "Accessories":
                text = "Accessories: None"
                color = (0, 255, 0)  # Always green
            elif label == "mask":
                if current_frame_labels['mask_correct']:
                    text = "Mask"
                    color = (0, 255, 0)  # Green for correct mask
                elif current_frame_labels['mask_incorrect']:
                    text = "Mask"
                    color = (0, 255, 255)  # Yellow for incorrect mask
                else:
                    text = "Mask"
                    color = (0, 0, 255)  # Red for no mask
            else:
                text = label.capitalize()
                color = (0, 255, 0) if current_frame_labels[label] else (0, 0, 255)

            cv2.putText(frame, text, (person_box[0], y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, lineType=cv2.LINE_AA)
            y_offset -= 30

    # Write the frame to the output video
    out.write(frame)

# Cleanup
cap.release()
out.release()
os.remove(temp_file.name)
cv2.destroyAllWindows()
