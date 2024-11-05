import os
import sys


class Coordinates:
    def __init__(self, x_left, y_bottom, x_right, y_top) -> None:
        self.x_left = x_left
        self.y_bottom = y_bottom
        self.x_right = x_right
        self.y_top = y_top