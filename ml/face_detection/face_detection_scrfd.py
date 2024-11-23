import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from typing import List, Tuple, Dict
import insightface

from app.utils import image_utils
from app.utils.data_access import DataAccessImage
from app.models.asset import Coordinates

SCRFD_THRESHOLD = 0.2


class SCRFDModel:
    def __init__(self, model_path, debug: bool = False, scrfd_threshold: float=0.2, create_margin: bool=False):
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

    def get_faces(self, data_layer: DataAccessImage, retry_count: int=0) -> Tuple[Dict, List, np.ndarray, Dict]:
        bboxes, landmarks = self.scrfd.detect(img=data_layer.get_bgr_image())
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
            raw_bboxes[ix] = {'coordinates': coordinates.__dict__, 'probability': round(probability, 4)}
            # logger.info(f"Old landmarks: {landmarks[ix]}")
            if probability >= self.scrfd_threshold:
                margined_coordinates = image_utils.create_margin(image=img, coordinates=coordinates, margin_scale=margin_scale)
                face_info = margined_coordinates.__dict__
                probabilities.append(probability)
                face_dict[ix] = face_info
                for key, value in face_dict[ix].items():
                    if value < 0:
                        face_dict[ix][key] = 0
                for lmk in landmarks[ix]:
                    lmk[0] = int(lmk[0])
                    lmk[1] = int(lmk[1])
                
                print(f"Face detection retry count: {retry_count}")
                if retry_count > 1:
                    return face_dict, probabilities, landmarks, raw_bboxes
                angle = self.compute_rotation_angle(original_points=landmarks[ix])
                print(f"Rotational angle: {angle}")
                img2 = data_layer.get_bgr_image()
                x, y = landmarks[ix][2]
                img2 = self.rotate_image(image=img2, angle_deg=-angle, center=(x, y))
                data_layer.image = img2
                if abs(angle) >= 50:
                    return self.get_faces(data_layer=data_layer, retry_count=retry_count + 1)
               
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
            # print(face_dict, probability, landmarks, raw_bboxes)
            imgs_raw_bboxes.append(raw_bboxes)
            face_dicts.append(face_dict)
            num_faces.append(len(face_dict))
            probabilities.append(probability)
            keypoints.append(landmarks)
        meta['keypoints'] = keypoints
        meta['probabilities'] = probabilities
        meta['num_faces'] = num_faces
        return face_dicts, keypoints, probabilities, imgs_raw_bboxes
    
    def set_detection_threshold(self, threshold: float) -> None:
        self.scrfd.det_thresh = threshold
        

if __name__ == "__main__":
    scrfd = SCRFDModel(model_path="/Users/divyanshnew/Documents/open_src_github/Identity-Management-Backend/ml/vision_models/face_detection/scrfd_10g_gnkps.onnx", create_margin=True)
    face_dicts, keypoints, probabilities, imgs_raw_bboxes = scrfd.get_faces_batch(data_access_frames=[DataAccessImage(file_path='/Users/divyanshnew/Downloads/image_2.jpeg')])
    print(face_dicts, keypoints, probabilities, imgs_raw_bboxes)