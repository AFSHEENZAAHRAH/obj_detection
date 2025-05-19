import sys
sys.path.append(r'C:\Users\afshe\OneDrive - Kumaraguru College of Technology\Desktop\obj_detect')  # Update this path
import cv2
import numpy as np
import torch
from lanenet_model.lanenet import LaneNet  # Ensure LaneNet is set up correctly
from yolov5.models.common import DetectMultiBackend  # Ensure YOLOv5 repo is set up correctly
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device