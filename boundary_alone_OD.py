import cv2
import torch
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  


cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Perform inference on the frame
    results = model(frame)
    
    # Extract detection results
    detections = results.pandas().xyxy[0]  # Get detections as a pandas DataFrame

    # Draw boundaries on detected objects
    for index, row in detections.iterrows():
        x_min, y_min, x_max, y_max, confidence, class_id, label = (
            int(row['xmin']),
            int(row['ymin']),
            int(row['xmax']),
            int(row['ymax']),
            row['confidence'],
            int(row['class']),
            row['name'],
        )
        # Draw a rectangle (boundary) around detected object
        color = (0, 0, 255)  # Red boundary for obstacles
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Highlight "avoid areas" as a red translucent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        alpha = 0.3  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Display the frame with detections and boundaries
    cv2.imshow('Autonomous Vehicle Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
