import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import types

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import scipy.io as sio
from math import cos, sin

from backbone_nets import mobilenetv2_backbone as backbone
# from backbone_nets import ghostnet_backbone

"""
need faceboxes
for face in facebox_output
enlarge box
image_crop
resize
torch from numpy -> convert to c, h, w format
normalize -> range : -127 to 128

forward_test: 
    - i2p.forward_test, return 3d attr and avgpool
    - i2p default in mobilenet_v2 (not pretrained shayad) : recheck
"""

class I2P(nn.Module):
	def __init__(self):
		super(I2P, self).__init__()
		self.backbone = getattr(backbone, 'mobilenet_v2')(pretrained=False)
		# print(self.backbone)

	def forward(self,input, target):
		"""Training time forward"""
		_3D_attr, avgpool = self.backbone(input)
		_3D_attr_GT = target.type(torch.cuda.FloatTensor)
		return _3D_attr, _3D_attr_GT, avgpool

	def forward_test(self, input):
		""" Testing time forward."""
		_3D_attr, avgpool = self.backbone(input)
		return _3D_attr, avgpool
	
I2P()