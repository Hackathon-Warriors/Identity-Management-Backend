import os
import sys
sys.path.append(os.getcwd())
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch import Tensor
import threading


from ml.face_detection.blazeface import BlazeFace

from app.utils import image_utils
from app.utils.data_access import DataAccessImage
from app.models.asset import Coordinates

BLAZEFACE_BACK_THRESHOLD = 0.6


class BlazeeFaceThales:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BlazeeFaceThales, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, weights_path, anchors_path, debug: bool = False, device: str = 'cpu'):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.blazeface = BlazeFace(back_model=True).to(device)
        self.blazeface.load_weights(weights_path)
        self.blazeface.load_anchors(anchors_path)
        self.debug = debug

    def _display_frame(
        self,
        img: np.ndarray,
        total_detected_faces: int = None,
    ):
        # only for debugging purpose
        if self.debug:
            msg = f"total_detected_faces: {total_detected_faces}"
            cv2.imshow(msg, img)
            waiting_time_in_ms = 400
            if cv2.waitKey(waiting_time_in_ms) or 0xFF == ord("q"):
                cv2.destroyAllWindows()
                
    def draw_bounding_boxes(self, img: np.ndarray, face_dict: Dict):
        for i, face_info in face_dict.items():
            xmin = int(face_info['x_left'])
            ymin = int(face_info['y_bottom'])
            xmax = int(face_info['x_right'])
            ymax = int(face_info['y_top'])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_margin(self, image, coordinates: Coordinates, margin_scale: float=0) -> Coordinates:
        height = abs(coordinates.y_top - coordinates.y_bottom)
        width = abs(coordinates.x_right - coordinates.x_left)
        height_margin = margin_scale*height/2
        top_margin = 3*height_margin
        bottom_margin = 0
        if image.shape[0] / image.shape[1] > 1.2:
            width_margin = margin_scale*width
        else:
            width_margin = margin_scale*width/2
        # width_margin = margin_scale*width/2
        x_left = int(max(coordinates.x_left - width_margin, 0))
        y_bottom = int(max(coordinates.y_bottom - top_margin, 0))
        x_right = int(min(coordinates.x_right + width_margin, image.shape[1]))
        y_top = int(min(coordinates.y_top + bottom_margin, image.shape[0]))
        return Coordinates(x_left=x_left, x_right=x_right, y_bottom=y_bottom, y_top=y_top)
    
    
    def get_faces(self, detections: Tensor, probabilities: List[int], original_img: DataAccessImage) -> Dict:
        face_dict = {}
        curr_image = original_img.get_bgr_image()
        original_height, original_width = curr_image.shape[:2]
        
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        
        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)
        margin_scale = 0.2
            
        for i in range(detections.shape[0]):
            if probabilities[i] >= BLAZEFACE_BACK_THRESHOLD:
                ymin = round(detections[i, 0] * original_height, 2)
                xmin = round(detections[i, 1] * original_width, 2)
                ymax = round(detections[i, 2] * original_height, 2)
                xmax = round(detections[i, 3] * original_width, 2)
                
                coordinates = Coordinates(x_left=xmin, x_right=xmax, y_bottom=ymin, y_top=ymax)
                margined_coordinates = self.create_margin(image=curr_image, coordinates=coordinates, margin_scale=margin_scale)
                face_info = margined_coordinates.__dict__
                face_dict[i] = face_info
                if self.debug:
                    self.draw_bounding_boxes(curr_image, face_dict)
        return face_dict
    
    def get_faces_batch(self, data_access_frames: List[DataAccessImage], meta: Dict=None) -> Tuple[List[Dict], List[int]]:
        if meta is None:
            meta = {}
            
        xback = np.zeros((len(data_access_frames), 256, 256, 3), dtype=np.uint8)
        
        for i, frame in enumerate(data_access_frames):
            img = frame.get_rgb_image()
            xback[i] = cv2.resize(img, (256, 256))
        back_detections = self.blazeface.predict_on_batch(x=xback, device=self.device)
        probabilities = [[face[-1].tolist() for face in frame] for frame in back_detections]
        
        face_dicts = []
        num_faces = []
        
        for i in range(len(data_access_frames)):
            face_dicts.append(self.get_faces(back_detections[i], probabilities[i], data_access_frames[i])) 
            num_faces.append(len(face_dicts[i]))   
        meta['probabilities'] = probabilities
        torch.cuda.empty_cache()
        return face_dicts, num_faces


if __name__ == "__main__":
    pass