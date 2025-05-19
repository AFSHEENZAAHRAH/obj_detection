import cv2
import torch
import numpy as np
import time

# Bayesian Filter (Kalman Filter)
class KalmanFilter:
    def __init__(self):
        # Initialize state (x, y, vx, vy), uncertainty, and state transition matrix
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # High uncertainty
        self.F = np.array([[1, 0, 1, 0],  # State transition matrix
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],  # Observation matrix
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 10  # Measurement noise covariance
        self.Q = np.eye(4) * 0.01  # Process noise covariance

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.state)  # Innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        return self.state[:2]  # Return x, y position

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5m' or 'yolov5l' for better accuracy

# Start webcam capture
cap = cv2.VideoCapture(0)  # 0 is the default webcam. Adjust if using a different device.

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Initialize Kalman Filter for tracking
kf = KalmanFilter()

# Define Occupancy Grid (2D Grid for simplicity)
occupancy_grid = np.zeros((480, 640), dtype=np.uint8)  # Grid size corresponds to image resolution

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
        # Update Occupancy Grid for detected objects
        occupancy_grid[y_min:y_max, x_min:x_max] = 255  # Mark occupied areas

        # Kalman Filter: predict and update with current detection
        kf.predict()
        measurement = np.array([x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2])
        kf.update(measurement)
        tracked_position = kf.get_state()
        
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
    
    # Display the occupancy grid (2D map of detected obstacles)
    cv2.imshow('Occupancy Grid', occupancy_grid)

    # Display the frame with detections and boundaries
    cv2.imshow('Autonomous Vehicle Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
