import os
import sys
sys.path.append(os.getcwd())

import time
import enum
from typing import List, Tuple, Dict
import cv2
import threading

from app.utils import image_utils
from app.utils.data_access import DataAccessImage
from app.models.asset import Coordinates
from ml.headpose_estimation.SynergyNet.synergy_thales import SynergyNet 

MIN_YAW = -30
MIN_PITCH = -40
MIN_ROLL = -26
MAX_YAW = 30
MAX_PITCH = 40
MAX_ROLL = 26

class HeadposeDirections(enum.Enum):
    LOOKING_RIGHT = "looking_right"
    LOOKING_LEFT = "looking_left"
    LOOKING_UPWARDS = "looking_upwards"
    LOOKING_DOWNWARDS = "looking_downwards"
    LOOKING_STRAIGHT = "looking_straight"


class HeadPoseEstimation:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HeadPoseEstimation, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str, debug: bool=False):
        self.hpes = SynergyNet(checkpoint_file_path=model_path, device='cpu')
        self.debug: bool = debug
        self.frame_wise_face_pose = []

    def get_orientation_msg(self, yaw, pitch, roll) -> str:
        orientation_message = None
        if (yaw and pitch and roll) is not None:
            if yaw > MAX_YAW:
                orientation_message = HeadposeDirections.LOOKING_RIGHT.value
            elif yaw < MIN_YAW:
                orientation_message = HeadposeDirections.LOOKING_LEFT.value
            elif pitch > MAX_PITCH:
                orientation_message = HeadposeDirections.LOOKING_UPWARDS.value
            elif pitch < MIN_PITCH:
                orientation_message = HeadposeDirections.LOOKING_DOWNWARDS.value
            
            if orientation_message is None:
                orientation_message = HeadposeDirections.LOOKING_STRAIGHT.value
        return orientation_message
    
    def get_orientation(
        self, data_layer: DataAccessImage, face_dict: Dict, largest_face_id
    ) -> str:
        pitch, yaw, roll = self.get_yaw_pitch_roll(data_layer=data_layer, face_dict=face_dict, largest_face_id=largest_face_id)
        orientation = self.get_orientation_msg(yaw=yaw, pitch=pitch, roll=roll)
        return orientation

    def get_yaw_pitch_roll(self, data_layer: DataAccessImage, face_dict: Dict, largest_face_id) -> Tuple[float, float, float]:
        pitch, yaw, roll = self.hpes.get_pose_v3(data_layer.get_bgr_image(), face_dict=face_dict, largest_face_id=largest_face_id)
        return pitch, yaw, roll


    def get_head_pose(
        self,
        image_batch: List[DataAccessImage],
        face_dicts: List[Dict],
        return_raw: bool = False,
    ) -> bool:
        orientations = []
        largest_face_ids = image_utils.largest_face_ids(face_dicts=face_dicts)
        for ix, data_layer in enumerate(image_batch):
            largest_id = largest_face_ids.get(ix)
            if largest_id is not None:
                orientation = self.get_orientation(
                    data_layer=data_layer,
                    face_dict=face_dicts[ix],
                    largest_face_id=largest_id,
                )
            else:
                orientation = None
            orientations.append(orientation)
        if orientations[0] == HeadposeDirections.LOOKING_STRAIGHT.value:
            return True
        return False