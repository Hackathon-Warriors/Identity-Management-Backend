import os
import sys
sys.path.append(os.getcwd())

import io
import json
import base64
import time
import cv2
import gc
import numpy as np
import traceback
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import insightface

# from utils import profiler, file_utils
from backend.utils import newrelic_utils
from backend.models import errors
from backend.data_access.data_access import DataAccessImage
from backend.models.asset import Coordinates
from utils import vision_logger, image_utils, functional_utils, viz_utils, profiler
from internal.vision_models import model_paths
from internal.face_match.face_match_params import SCRFD_THRESHOLD, ARCFACE_THRESHOLD
# from vision_utils.env_utils import DevMode, DEVELOPMENT_MODE

logger = vision_logger.VisionLogger(__name__)

class SCRFDModel:
    def __init__(self, model_path, debug: bool = False, scrfd_threshold: float=0.8, create_margin: bool=False):
        # self.scrfd = insightface.model_zoo.get_model(name=model_path, providers=['CoreMLExecutionProvider'])
        self.scrfd = insightface.model_zoo.get_model(name=model_path, providers=['CPUExecutionProvider'])
        self.scrfd.prepare(ctx_id=0, input_size=(640, 640))
        self.debug = debug
        self.scrfd_threshold = scrfd_threshold
        self.create_margin = create_margin
    
    def compute_rotation_angle(self, original_points):
        # Define canonical landmarks for a frontal face
        canonical_points = np.float32([
            [30, 30],  # Left eye
            [70, 30],  # Right eye
            [50, 50],  # Nose tip
            [30, 70],  # Left mouth corner
            [70, 70]   # Right mouth corner
        ])
        
        # Check if eyes are below the mouth
        # if original_points[0][1] > original_points[3][1] or original_points[1][1] > original_points[4][1]:
        #     return 180  # The image is at least 180 degrees flipped

        # Compute direction vectors
        orig_dir = original_points[1] - original_points[0]
        canonical_dir = canonical_points[1] - canonical_points[0]
        
        # Compute the cosine of the angle using the dot product formula
        cosine_angle = np.dot(orig_dir, canonical_dir) / (np.linalg.norm(orig_dir) * np.linalg.norm(canonical_dir))
        
        # Clamp the value between -1 and 1 to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Compute the angle in radians
        angle_rad = np.arccos(cosine_angle)
        
        # Determine the sign of the angle (clockwise or anti-clockwise)
        # Using the cross product's z-component
        cross_z = orig_dir[0] * canonical_dir[1] - orig_dir[1] * canonical_dir[0]
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        # If cross_z is negative, the rotation is clockwise, so we negate the angle
        if cross_z < 0:
            angle_deg = -angle_deg

        return angle_deg

    
    def rotate_point(self, image, x, y, angle_deg, center=None):
        if center is None:
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2
        else:
            center_x, center_y = center
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)
        
        # Translate the point to the origin
        x -= center_x
        y -= center_y
        
        # Apply the rotation matrix
        x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Translate the point back
        x_new += center_x
        y_new += center_y
        
        return int(x_new), int(y_new) 
    
    def rotate_image(self, image, angle_deg, center=None):
        # If no rotation center is specified, use the center of the image
        if center is None:
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2
        else:
            center_x, center_y = center

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1)

        # Rotate the image
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        return rotated_image

    @profiler.memory_profiler
    def get_faces(self, data_layer: DataAccessImage, retry_count: int=0) -> Tuple[Dict, List, np.ndarray, Dict]:
        bboxes, landmarks = self.scrfd.detect(img=data_layer.get_bgr_image())
        logger.info(f"Scrfd Raw bboxes, probability: {bboxes} and landmarks: {landmarks}")
        img = data_layer.get_bgr_image()
        raw_bboxes = {}
        face_dict = {}
        probabilities = []
        margin_scale = 0
        if self.create_margin:
            margin_scale = 0.1
        for ix, bbox in enumerate(bboxes):
            x_left, y_bottom, x_right, y_top, probability = bbox
            coordinates = Coordinates(x_left=int(x_left), 
                                      x_right=int(x_right), 
                                      y_bottom=int(y_bottom), 
                                      y_top=int(y_top))
            # logger.info(f"Old face coords: {coordinates.__dict__}\n")
            raw_bboxes[ix] = {'coordinates': coordinates.__dict__, 'probability': functional_utils.round_float(probability, places=4)}
            # logger.info(f"Old landmarks: {landmarks[ix]}")
            if probability >= self.scrfd_threshold:
                margined_coordinates = image_utils.create_margin(image=img, coordinates=coordinates, margin_scale=margin_scale)
                face_info = margined_coordinates.__dict__
                probabilities.append(probability)
                face_dict[ix] = face_info
                for key, value in face_dict[ix].items():
                    if value < 0:
                        face_dict[ix][key] = 0
                logger.info(f"Face dict {ix}: {face_dict[ix]}, probability: {probability}")
                for lmk in landmarks[ix]:
                    lmk[0] = int(lmk[0])
                    lmk[1] = int(lmk[1])
                if self.debug:
                    img = image_utils.draw_on_image(image=img, **face_info)
                    for lmk in landmarks[ix]:
                        x=lmk[0]
                        y=lmk[1]
                        cv2.circle(img=img, center=(int(x), int(y)), radius=4, color=(0, 0, 255), thickness=5)
                        # cv2.putText(img, f"{(int(x), int(y))}", (int(x), int(y)), 1, 0.1, (0, 0, 255), 1, cv2.LINE_AA) 
                        cv2.putText(img, f"{(int(x), int(y))}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)         
                        image_utils.show_image(img)
                
                logger.info(f"Face detection retry count: {retry_count}")
                if retry_count > 1:
                    return face_dict, probabilities, landmarks, raw_bboxes
                angle = self.compute_rotation_angle(original_points=landmarks[ix])
                logger.info(f"Rotational angle: {angle}")
                img2 = data_layer.get_bgr_image()
                x, y = landmarks[ix][2]
                img2 = self.rotate_image(image=img2, angle_deg=-angle, center=(x, y))
                data_layer.image = img2
                if abs(angle) >= 50:
                    return self.get_faces(data_layer=data_layer, retry_count=retry_count + 1)
                # for lmk in landmarks[ix]:
                #     lmk[0], lmk[1] = self.rotate_point(image=img2, x=lmk[0], y=lmk[1], angle_deg=-angle, center=(x, y))
                #     if self.debug:
                #         cv2.circle(img=img2, center=(int(lmk[0]), int(lmk[1])), radius=4, color=(0, 0, 255), thickness=5)
                # margined_coordinates.x_left, margined_coordinates.y_bottom = self.rotate_point(image=img2, x=margined_coordinates.x_left, y=margined_coordinates.y_bottom, angle_deg=-angle, center=(x, y))
                # margined_coordinates.x_right, margined_coordinates.y_top = self.rotate_point(image=img2, x=margined_coordinates.x_right, y=margined_coordinates.y_top, angle_deg=-angle, center=(x, y))
                # d = margined_coordinates.__dict__
                # if self.debug:
                #     img2 = image_utils.draw_on_image(image=img2, **d)
                #     image_utils.show_image(img2)
                # face_dict[ix] = margined_coordinates.__dict__
        return face_dict, probabilities, landmarks, raw_bboxes
    
    def get_faces_batch(self, data_access_frames: List[DataAccessImage], meta: Dict=None) -> Tuple[List[Dict], List[np.ndarray], List[List], List[Dict]]:
        face_dicts = []
        num_faces = []
        probabilities = []
        keypoints = []
        imgs_raw_bboxes = []
        if meta is None:
            meta = {}
        for ix, data_layers in enumerate(data_access_frames):
            face_dict, probability, landmarks, raw_bboxes = self.get_faces(data_layer=data_layers)
            imgs_raw_bboxes.append(raw_bboxes)
            face_dicts.append(face_dict)
            num_faces.append(len(face_dict))
            probabilities.append(probability)
            keypoints.append(landmarks)
        meta['keypoints'] = keypoints
        meta['probabilities'] = probabilities
        meta['num_faces'] = num_faces
        logger.info(f"meta:{meta}\nface_dicts: {face_dicts}\nprobabilities: {probabilities}")
        return face_dicts, keypoints, probabilities, imgs_raw_bboxes
    
    def set_detection_threshold(self, threshold: float) -> None:
        self.scrfd.det_thresh = threshold
        

if __name__ == "__main__":
    detector_model = SCRFDModel(model_path=model_paths.SCRFD_DETECTION, debug=True, scrfd_threshold=SCRFD_THRESHOLD, create_margin=True)
    
    local_file_path = "/Users/vedant.bajaj/singleFaceDetection/images/kys_cases/case8/8_img_1.png"
    img = DataAccessImage(file_path=local_file_path)
    
    face_dicts, keypoints, probabilities, imgs_raw_bboxes = detector_model.get_faces_batch([img])
    print(f"Face Dict: {face_dicts}, landmarks: {keypoints}, probabilities: {probabilities}")
    print(f"\n imgs_raw_bboxes: {imgs_raw_bboxes}")