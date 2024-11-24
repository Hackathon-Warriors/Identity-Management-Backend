import os
import sys
sys.path.append(os.getcwd())
from typing import List, Dict, Tuple
import threading


from app.utils import image_utils
from app.utils.data_access import DataAccessImage
from app.models.asset import Coordinates


import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MEDIAPIPE_LEFT_EYE_BLINK_THRESH = 0.5392706394195557
MEDIAPIPE_RIGHT_EYE_BLINK_THRESH = 0.5787549018859863

# https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/FaceLandmarkerOptions

class EyesOpenMediaPipe:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EyesOpenMediaPipe, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str) -> None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            num_faces=1,
                                            min_face_detection_confidence= 0.1,
                                            min_face_presence_confidence= 0.1,
                                            min_tracking_confidence= 0.1,
                                            running_mode=mp.tasks.vision.RunningMode.IMAGE)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def convert_frame_to_mp_image(self, frame):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    def inference(self, data_access_frames: List[DataAccessImage]):
        blends = []
        results = [False]*len(data_access_frames)
        for ix, data_layer in enumerate(data_access_frames):
            frame = data_layer.get_bgr_image()
            mp_image = self.convert_frame_to_mp_image(frame)
            result = self.detector.detect(mp_image)
            points = {'frame_id': data_layer.frame_id}

            # print(f"Result: {result}")
            if len(result.face_blendshapes)>0:
                face_blendshapes = result.face_blendshapes[0]
                for face_blendshapes_category in face_blendshapes:
                    # if face_blendshapes_category.category_name in ('eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight'):
                    if face_blendshapes_category.category_name in ('eyeBlinkLeft', 'eyeBlinkRight'):
                        points[face_blendshapes_category.category_name] = round(1 - face_blendshapes_category.score, 4)
            blends.append(points)
        
        for ix, blend in enumerate(blends):
            left_open = False
            right_open = False
            eyes_open = False
            if blend.get('eyeBlinkLeft', 0) >= MEDIAPIPE_LEFT_EYE_BLINK_THRESH:
                left_open = True
            if blend.get('eyeBlinkRight', 0) >= MEDIAPIPE_RIGHT_EYE_BLINK_THRESH:
                right_open = True
            eyes_open = left_open or right_open
            results[ix]=eyes_open
        return blends, results
    
    
    def eyes_open_batch(self, data_layer_batch: List[DataAccessImage], faces: List[Dict], **kwargs) -> bool:
        meta = {}
        are_eyes_open = [False] * len(data_layer_batch)
        largest_face_ids = image_utils.largest_face_ids(face_dicts=faces)
        aggregate_response = False
        index_mapping_cropped_faces = {}
        cropped_faces = []
        for ix, (data_layer, face_dict) in enumerate(zip(data_layer_batch, faces)):
            largest_id = largest_face_ids.get(ix)
            if largest_id is not None:
                face = None
                face = Coordinates(**face_dict[largest_id])
                if face:
                    image = data_layer.get_bgr_image()
                    face_image = image[
                        face.y_bottom : face.y_top, face.x_left : face.x_right
                    ]
                    new_data_layer = DataAccessImage(image=face_image, frame_id=data_layer.frame_id)
                    cropped_faces.append(new_data_layer)
                    index_mapping_cropped_faces[ix] = len(cropped_faces) - 1
        if len(cropped_faces) > 0:
            blends, response = self.inference(data_access_frames=cropped_faces)
            meta.update({'eye_open_blends': blends})
            for ix in range(len(are_eyes_open)):
                if index_mapping_cropped_faces.get(ix) is not None:
                    is_frame_eyes_open = response[index_mapping_cropped_faces.get(ix)]
                    are_eyes_open[ix] = is_frame_eyes_open
        if are_eyes_open[0] == True:
            return True
        return False    

if __name__ == "__main__":
    pass