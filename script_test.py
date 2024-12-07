from datetime import datetime
from ultralytics import YOLO
import cv2
import math
import os

def video_detection(path_x, output_path):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("YOLO-Weights/best_test_150.pt")
    
    # Update class names to only include the target classes.
    classNames = ['hairnet', 'helmet', 'mask', 'person', 'safety vest']
    
    # Define classes to detect (Updated to include only the specified classes)
    target_classes = ['hairnet', 'helmet', 'mask', 'person', 'safety vest']
    
    # Initialize variables
    start_time = datetime.now()
    detection_results = []
    person_box = None  # To store the position of the "Person" bounding box
    label_offset = 10  # Vertical offset for labels
    colors = {
        'detected': (0, 255, 0),  # Green for detected classes
        'not_detected': (0, 0, 255)  # Red for not detected classes
    }

    # Initialize VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))  # 30 FPS output video

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit the loop if the video ends

        results = model(img, stream=True)  # Results will now be a generator

        # Reset detected labels for the current frame
        current_frame_labels = {label: False for label in target_classes}  # Set all to False initially
        detected_labels = set()  # Reset detected labels for each frame
        person_box = None  # Reset the person box for each frame

        for result in results:  # Iterate through the generator results
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Only consider target classes and confidence threshold
                if class_name not in target_classes or conf <= 0.5:
                    continue

                # Mark as detected for the current class
                if class_name in current_frame_labels:
                    current_frame_labels[class_name] = True

                # If we detect a "Person", store the bounding box to draw it once
                if class_name == "person" and conf > 0.6:
                    person_box = (x1, y1, x2, y2)

        # Check if person_box is None before drawing
        if person_box is not None:  # Only draw if "Person" is detected
            cv2.rectangle(img, (person_box[0], person_box[1]), (person_box[2], person_box[3]), (0, 255, 0), 2)

            # Draw labels for the target classes (including Hairnet)
            y_offset = person_box[1] - label_offset  # Offset for labels based on "Person" position
            for label in target_classes:
                if label != "person" and label in current_frame_labels:
                    label_text = label.replace("NO-", "")
                    label_color = colors['detected'] if current_frame_labels[label] else colors['not_detected']  # Green if detected, Red if not
                    t_size = cv2.getTextSize(label_text, 0, fontScale=1, thickness=2)[0]

                    # Stack the labels vertically with an offset for each label
                    y_offset -= t_size[1] + 2  # Adjust the space between labels
                    cv2.putText(img, label_text, (person_box[0], y_offset), 0, 1, label_color, thickness=2, lineType=cv2.LINE_AA)

                    # Mark the label as detected and avoid repeating it
                    detected_labels.add(label)

        # Write the frame to the output video
        out.write(img)

        # Yield the processed frame for display (optional)
        yield img

        # Reset detection results every 30 seconds (if required)
        if (datetime.now() - start_time).seconds >= 30:
            with open('detection_results.txt', 'a') as file:
                for detection in detection_results:
                    file.write(f"[ {detection['time']} ] {detection['class']} {detection['confidence']} {detection['bounding_box']} \n")
                file.write('\n')  # Add a newline to separate each 30-second interval
            start_time = datetime.now()
            detection_results = []

    # Release the VideoWriter and capture objects
    cap.release()
    out.release()

# Call the video_detection function with your video path and output path
video_path = r"C:\Users\acer\SMF\orj_1.mp4"  # Make sure to use raw string or escape backslashes
output_path = os.path.join(os.path.dirname(video_path), "CV_out_test.mp4")  # Save the output video in the same directory as the input video

# Process the video frames
for img in video_detection(video_path, output_path):
    # You can display the frames, but now the video is being saved as well
    cv2.imshow("Detection Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop the video
        break

cv2.destroyAllWindows()
