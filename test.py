from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("YOLO-Weights/best.pt")

# Open webcam (0 usually corresponds to the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Perform inference on the current frame
    results = model(frame)
    
    # Results are in 'results.xywh' format
    # Draw boxes and labels on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID
        label = model.names[cls]  # Class name
        
        # Draw bounding box and label on the frame
        color = (0, 255, 0)  # Green for detected objects
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        frame = cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detections
    cv2.imshow("Webcam Detection", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
