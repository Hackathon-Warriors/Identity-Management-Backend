import os
import sys
sys.path.append(os.getcwd())

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