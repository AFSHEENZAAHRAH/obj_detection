import cv2
import torch
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  


cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()


def calculate_grid_indices(x1, y1, grid_size, frame_width, frame_height):
    cell_width = frame_width // grid_size[1]
    cell_height = frame_height // grid_size[0]
    grid_col = x1 // cell_width
    grid_row = y1 // cell_height
    return grid_row, grid_col


grid_size = (5, 5)  

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    
    frame_height, frame_width, _ = frame.shape

    
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  

    
    frame_with_boxes = np.array(results.render()[0]) 

    cell_height = frame_height // grid_size[0]
    cell_width = frame_width // grid_size[1]
    for i in range(1, grid_size[0]):
        y = i * cell_height
        cv2.line(frame_with_boxes, (0, y), (frame_width, y), (255, 255, 255), 1)
    for j in range(1, grid_size[1]):
        x = j * cell_width
        cv2.line(frame_with_boxes, (x, 0), (x, frame_height), (255, 255, 255), 1)

    
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
       
        grid_row, grid_col = calculate_grid_indices(x1, y1, grid_size, frame_width, frame_height)
        label = f"{model.names[cls]} ({grid_row},{grid_col})"
        cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   
    cv2.imshow('YOLOv5 Object Detection with Grid', frame_with_boxes)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
