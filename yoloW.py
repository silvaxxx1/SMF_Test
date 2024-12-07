import cv2
import tempfile
import time
from inference_sdk import InferenceHTTPClient
import os

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VN28ceooZnimGyMKzkNo"
)

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

# Create a VideoWriter object to save the output video in the same directory
output_video = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video, fourcc, fps, (640, 480))

# Loop through frames of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame (optional)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480 or any other size

    # Save the frame to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, frame_resized)
        
        # Perform inference on the image file by providing the file path directly
        predictions = CLIENT.infer(temp_file_path, model_id="smart-meta-factory/1")
        
        # Process the predictions and draw bounding boxes
        if 'predictions' in predictions:
            for prediction in predictions['predictions']:
                x = int(prediction['x'])
                y = int(prediction['y'])
                w = int(prediction['width'])
                h = int(prediction['height'])
                label = prediction['class']
                confidence = prediction['confidence']
                
                # Draw bounding box
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{label} ({confidence*100:.1f}%)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame_resized)

    # Give the system some time to release the lock on the file
    time.sleep(0.5)  # Adjust the sleep time if necessary

    # Now attempt to delete the temporary file after processing
    try:
        os.remove(temp_file_path)
    except PermissionError:
        print(f"Error: Could not delete the temporary file {temp_file_path}")

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
