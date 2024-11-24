import os
import sys
sys.path.append(os.getcwd())
from typing import List, Dict

from app.models.asset import Coordinates


def create_margin(image, coordinates: Coordinates, margin_scale: float=0) -> Coordinates:
    height = abs(coordinates.y_top - coordinates.y_bottom)
    width = abs(coordinates.x_right - coordinates.x_left)
    height_margin = margin_scale*height/2
    width_margin = margin_scale*width/2
    x_left = int(max(coordinates.x_left - width_margin, 0))
    y_bottom = int(max(coordinates.y_bottom - height_margin, 0))
    x_right = int(min(coordinates.x_right + width_margin, image.shape[1]))
    y_top = int(min(coordinates.y_top + height_margin, image.shape[0]))
    return Coordinates(x_left=x_left, x_right=x_right, y_bottom=y_bottom, y_top=y_top)


def largest_face_ids(face_dicts: List[Dict]) -> Dict:
    largest_face_id_dict = {}
    for frame_id, face_dict in enumerate(face_dicts):
        max_area = 0
        max_area_face_id = None # if no face detected in the image
        for face_id, face in face_dict.items():
            x_left, y_bottom, x_right, y_top = face['x_left'], face['y_bottom'], face['x_right'], face['y_top']
            width = x_right - x_left
            height = y_top - y_bottom
            area = width * height
            if area > max_area:
                max_area = area
                max_area_face_id = face_id
        largest_face_id_dict[frame_id] = max_area_face_id    
    return largest_face_id_dict