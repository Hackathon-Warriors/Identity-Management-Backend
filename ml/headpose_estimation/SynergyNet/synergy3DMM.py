import torch
import torch.nn as nn
import numpy as np
# from torchvision import transforms as T
import scipy.io as sio
from math import cos, sin


# All data parameters import
from utils.params import ParamsPack
param_pack = ParamsPack()

from backbone_nets import resnet_backbone
from backbone_nets import mobilenetv1_backbone
from backbone_nets import mobilenetv2_backbone
from backbone_nets import ghostnet_backbone
# from backbone_nets.pointnet_backbone import MLP_for, MLP_rev
import loss_definition
# from loss_definition import ParamLoss, WingLoss

from backbone_nets.ResNeSt import resnest50, resnest101
import time
from utils.inference import predict_sparseVert, predict_denseVert, predict_pose, crop_img
from FaceBoxes import FaceBoxes
import cv2
import types
import os

prefix_path = os.path.abspath(loss_definition.__file__).rsplit('/',1)[0]
print(prefix_path)

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
	def __init__(self, args):
		super(I2P, self).__init__()
		self.args = args
		# backbone definition
		if 'mobilenet_v2' in self.args.arch:
			self.backbone = getattr(mobilenetv2_backbone, args.arch)(pretrained=False)
		elif 'mobilenet' in self.args.arch:
			self.backbone = getattr(mobilenetv1_backbone, args.arch)()		
		elif 'resnet' in self.args.arch:
			self.backbone = getattr(resnet_backbone, args.arch)(pretrained=False)
		elif 'ghostnet' in self.args.arch:
			self.backbone = getattr(ghostnet_backbone, args.arch)()
		elif 'resnest' in self.args.arch:
			self.backbone = resnest50()
		else:
			raise RuntimeError("Please choose [mobilenet_v2, mobilenet_1, resnet50, or ghostnet]")

	def forward(self,input, target):
		"""Training time forward"""
		_3D_attr, avgpool = self.backbone(input)
		_3D_attr_GT = target.type(torch.cuda.FloatTensor)
		return _3D_attr, _3D_attr_GT, avgpool

	def forward_test(self, input):
		""" Testing time forward."""
		_3D_attr, avgpool = self.backbone(input)
		return _3D_attr, avgpool
        
# Main model SynergyNet definition
def show_image(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

class SynergyNet(nn.Module):
	def __init__(self):
		super(SynergyNet, self).__init__()
		self.triangles = sio.loadmat(os.path.join(prefix_path, '3dmm_data/tri.mat'))['tri'] -1
		self.triangles = torch.Tensor(self.triangles.astype(np.int64)).long()
		args = types.SimpleNamespace()
		args.arch = 'mobilenet_v2'
		args.checkpoint_fp = os.path.join(prefix_path, 'pretrained/best.pth.tar')

		# Image-to-parameter
		self.I2P = I2P(args)
		# Forward
		# self.forwardDirection = MLP_for(68)
		# Reverse
		# self.reverseDirection = MLP_rev(68)
		try:
			# print("loading weights from ", args.checkpoint_fp)
			self.load_weights(args.checkpoint_fp)
		except:
			pass
		self.eval()

	def forward_test(self, input):
		"""test time forward"""
		_3D_attr, _ = self.I2P.forward_test(input)
		return _3D_attr

	def load_weights(self, path):
		model_dict = self.state_dict()
		# print(f"Model dict: {model_dict}")
		checkpoint = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
		# print(f"Checkpoint: {checkpoint}")

		# because the model is trained by multiple gpus, prefix 'module' should be removed
		for k in checkpoint.keys():
			model_dict[k.replace('module.', '')] = checkpoint[k]

		self.load_state_dict(model_dict, strict=False)

	def get_pose(self, input):
		face_boxes = FaceBoxes()
		rects = face_boxes(input)
		pitch, yaw, roll = 0, 0, 0

		# storage
		pts_res = []
		poses = {
			'pitch': [],
			'yaw': [],
			'roll': []
		}
		vertices_lst = []
		for idx, rect in enumerate(rects):
			roi_box = rect

			# enlarge the bbox a little and do a square crop
			HCenter = (rect[1] + rect[3])/2
			WCenter = (rect[0] + rect[2])/2
			side_len = roi_box[3]-roi_box[1]
			margin = side_len * 1.2 // 2
			roi_box[0], roi_box[1], roi_box[2], roi_box[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin
			# show_image(input)
			img = crop_img(input, roi_box)
			# show_image(image=img)
			img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
			img = torch.from_numpy(img)
			img = img.permute(2,0,1)
			img = img.unsqueeze(0)
			img = (img - 127.5)/ 128.0

			with torch.no_grad():
				param = self.forward_test(img)
				# print(f"param i2p output: {param}")
			
			param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

			angles, translation = predict_pose(param, roi_box)
			# print(f"Angles: {angles}")
			yaw, pitch, roll = angles
			print(f"Pitch: {pitch}, yaw: {yaw}, roll: {roll}")

			# poses.append([angles, translation])
			# poses.append((pitch, yaw, roll))
			poses['pitch'].append(pitch)
			poses['yaw'].append(yaw)
			poses['roll'].append(roll)

		# return [pitch], [yaw], [roll]
		return poses['pitch'], poses['yaw'], poses['roll']
	
	def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
		"""
		Prints the person's name and age.

		If the argument 'additional' is passed, then it is appended after the main info.

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

		cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
		cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
		cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

		return img

	def get_all_outputs(self, input):
		"""convenient api to get 3d landmarks, face pose, 3d faces"""

		face_boxes = FaceBoxes()
		rects = face_boxes(input)

		# storage
		pts_res = []
		poses = []
		vertices_lst = []
		for idx, rect in enumerate(rects):
			roi_box = rect

			# enlarge the bbox a little and do a square crop
			HCenter = (rect[1] + rect[3])/2
			WCenter = (rect[0] + rect[2])/2
			side_len = roi_box[3]-roi_box[1]
			margin = side_len * 1.2 // 2
			roi_box[0], roi_box[1], roi_box[2], roi_box[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin
			show_image(input)
			img = crop_img(input, roi_box)
			show_image(image=img)
			img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
			img = torch.from_numpy(img)
			img = img.permute(2,0,1)
			img = img.unsqueeze(0)
			img = (img - 127.5)/ 128.0

			with torch.no_grad():
				param = self.forward_test(img)
			
			param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

			lmks = predict_sparseVert(param, roi_box, transform=True)
			vertices = predict_denseVert(param, roi_box,  transform=True)
			angles, translation = predict_pose(param, roi_box)
			# print(f"Angles: {angles}")
			yaw, pitch, roll = angles
			# print(f"Pitch: {pitch}, yaw: {yaw}, roll: {roll}")

			pts_res.append(lmks)
			vertices_lst.append(vertices)
			poses.append([angles, translation])

		return pts_res, vertices_lst, poses

if __name__ == '__main__':
	pass
