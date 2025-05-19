import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use 'yolov5m' or 'yolov5l' for better accuracy

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 is the default webcam. If you have multiple webcams, you can try 1, 2, etc.

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is not captured properly, exit
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Perform inference on the frame
    results = model(frame)  # Pass the frame to the YOLOv5 model

    # Render predictions on the frame
    frame = results.render()[0]  # Render detections on the frame

    # Display the frame with detections
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
