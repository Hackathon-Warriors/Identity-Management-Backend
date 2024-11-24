import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/Users/divyanshnew/Documents/open_src_github/Identity-Management-Backend/ml/headpose_estimation/SynergyNet/")

import cv2
from math import cos, sin
import numpy as np
from typing import Dict, List

import torch
import torch.nn as nn

from ml.headpose_estimation.SynergyNet.utils.params import ParamsPack
# import loss_definition
from ml.headpose_estimation.SynergyNet.utils.inference import (
    predict_sparseVert,
    # predict_pose,
    # crop_img,
    parse_pose
)
from backbone_nets import mobilenetv2_backbone

param_pack = ParamsPack()


class Coordinates:
    def __init__(self, x_left, y_bottom, x_right, y_top) -> None:
        self.x_left = x_left
        self.y_bottom = y_bottom
        self.x_right = x_right
        self.y_top = y_top


def parse_param_62(param):
    """Work for only tensor"""
    p_ = param[:, :12].reshape(-1, 3, 4)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(-1, 3, 1)
    alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
    alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
    return p, offset, alpha_shp, alpha_exp


# Image-to-parameter
class I2P(nn.Module):
    def __init__(self):
        super(I2P, self).__init__()
        self.backbone = getattr(mobilenetv2_backbone, "mobilenet_v2")(pretrained=False)

    def forward(self, input, target):
        """Training time forward"""
        _3D_attr, avgpool = self.backbone(input)
        _3D_attr_GT = target.type(torch.cuda.FloatTensor)
        return _3D_attr, _3D_attr_GT, avgpool

    def forward_test(self, input):
        """Testing time forward."""
        _3D_attr, avgpool = self.backbone(input)
        return _3D_attr, avgpool



class SynergyNet(nn.Module):
    def __init__(self, checkpoint_file_path: str=None, device: str='cpu'):
        super(SynergyNet, self).__init__()
        self.I2P = I2P()
        self.load_weights(checkpoint_file_path)
        self.device = device
        self.eval()

    def forward_test(self, input):
        """test time forward"""
        _3D_attr, _ = self.I2P.forward_test(input)
        return _3D_attr

    def load_weights(self, path):
        model_dict = self.state_dict()
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
        # because the model is trained by multiple gpus, prefix 'module' should be removed
        for k in checkpoint.keys():
            model_dict[k.replace("module.", "")] = checkpoint[k]
        self.load_state_dict(model_dict, strict=False)


    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        """
        Parameters
        ----------
        img : array
                Target image to be drawn on
        yaw : int
                yaw rotation
        pitch: int
                pitch rotation
        roll: int
                roll rotation
        tdx : int , optional
                shift on x axis
        tdy : int , optional
                shift on y axis

        Returns
        -------
        img : array
        """

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)

        return img
    
    def predict_pose(self, param):
        P, angles, t3d = parse_pose(param)
        return angles
    
    def get_pose_v3(self, image: np.ndarray, face_dict: Dict, largest_face_id: str):
        pitch, yaw, roll = None, None, None
        face = Coordinates(**face_dict[largest_face_id])
        img = image[face.y_bottom : face.y_top, face.x_left : face.x_right]
        img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = (img - 127.5) / 128.0

        with torch.no_grad():
            param = self.forward_test(img.to(self.device))
        

        # if torch.cuda.is_available():
        #     param = param.squeeze().numpy().flatten().astype(np.float32)
        # else:
        #     param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        torch.cuda.empty_cache()

        angles = self.predict_pose(param)
        yaw, pitch, roll = angles
        return pitch, yaw, roll
    

    def get_landmarks(self, image: np.ndarray, face_dict: Dict):
        landmarks = {}
        for face_no in face_dict.keys():
            face = Coordinates(**face_dict[face_no])
            img = image[face.y_bottom : face.y_top, face.x_left : face.x_right]
            img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = (img - 127.5) / 128.0
            with torch.no_grad():
                param = self.forward_test(img)

            if torch.cuda.is_available():
                param = param.squeeze().numpy().flatten().astype(np.float32)
            else:
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            
            roi_box = [face.x_left, face.y_bottom, face.x_right, face.y_top, 1]
            lmks = predict_sparseVert(param=param, roi_box=roi_box, transform=True)
            landmarks[face_no] = lmks
        return landmarks

if __name__ == "__main__":
    pass