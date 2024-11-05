import os
import sys

sys.path.append(os.getcwd())
import traceback

import numpy as np
from typing import Tuple
import cv2
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAccessImage:
    def __init__(
        self,
        file_path: str=None,
        delete_at_destruction: bool = None,
        image: np.ndarray = None,
        frame_id: int = None
    ) -> None:
        self.file_path = file_path
        self.delete_at_destruction = delete_at_destruction if delete_at_destruction is not None else False
        self.image = image
        self.pil_image = None
        self.frame_id = frame_id

    def __del__(self):
        if self.delete_at_destruction:
            print(f"Cleaning up file_path: {self.file_path}")
            file_utils.cleanup([self.file_path])

    def read_image_cv2(self, **kwargs):
        try:
            self.image = cv2.imread(self.file_path, **kwargs)
            if self.image is None:
                raise Exception(message=f"Bad image received")
        except Exception as e:
            raise e

        return self.image
    
    def read_image_pil(self, **kwargs):
        if self.pil_image is None:
            self.pil_image = Image.open(self.file_path, **kwargs)
        return self.pil_image.copy()
    
    def get_bgr_image(self) -> np.ndarray:
        if self.image is None:
            self.image = self.read_image_cv2()
        copy = self.image.copy()
        return copy
    
    def get_rgb_image(self) -> np.ndarray:
        if self.image is None:
            self.image = self.read_image_cv2()
        copy = self.image.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        return copy
    
    def get_grayscale_image(self) -> np.ndarray:
        if self.image is None:
            self.image = self.read_image_cv2()
        copy = self.image.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        return copy
    
    def get_dimension(self) -> Tuple[float, float]:
        if self.image is None:
            self.image = self.read_image_cv2()
        height, widht, channel = self.image.shape
        return (height, widht)


if __name__ == "__main__": 
    file_path = ''
    data_access_obj = DataAccessImage(file_path=file_path, delete_at_destruction=True)
    data_access_obj.get_rgb_image()