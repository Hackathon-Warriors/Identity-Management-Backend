import os
import sys
sys.path.append(os.getcwd())

import enum

from dataclasses import dataclass


class Coordinates:
    def __init__(self, x_left, y_bottom, x_right, y_top) -> None:
        self.x_left = x_left
        self.y_bottom = y_bottom
        self.x_right = x_right
        self.y_top = y_top


@dataclass
class InternalLivenessResponse:
    is_live: bool
    msg: str


class Messaging(enum.Enum):
    LIVE = "The user is live."
    MULTIPLE_FACES = "Multiple faces detected, please retry."
    NO_FACE = "No face detected, please retry."
    EYES_OPEN = "Please ensure your eyes are open."
    HEADPOSE = "Please ensure that you are looking straight."
    SPOOF = "Please ensure you are not clicking photo of photo, mask etc"



@dataclass
class FaceMatchResponse:
    is_similar: bool
    msg: str 