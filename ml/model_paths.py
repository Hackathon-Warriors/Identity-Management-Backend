import os
import sys
sys.path.append(os.getcwd())


BLAZEFAZE_BACK_WEIGHTS_PATH = 'ml/vision_models/face_detection/blazefaceback.pth'
BLAZEFAZE_BACK_ANCHORS_PATH =  'ml/vision_models/face_detection/anchorsback.npy'

SYNERGYNET_MODEL_PATH = 'ml/vision_models/headpose_estimation/synergynet_model.pth.tar'

EYES_OPEN_MODEL_PATH = "ml/vision_models/eyes_open/face_landmarker.task"

SPOOF_MODEL_PATH = 'ml/vision_models/spoof_detection/spoof_detection_eva_kaggle.pth'

FACE_MATCH_SCRFD = 'ml/vision_models/face_match/scrfd_10g_gnkps.onnx'
FACE_MATCH_EMBEDDING = 'ml/vision_models/face_match/arcface_r100_v1.onnx'