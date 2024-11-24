import os
import sys
sys.path.append(os.getcwd())

from dataclasses import asdict

from app.utils import image_utils
from app.models.asset import InternalLivenessResponse, Messaging
from app.utils.data_access import DataAccessImage
from app.models.asset import Coordinates

from ml.face_detection.blazeface_back_detection import BlazeeFaceThales
from ml.eyes_open.Mediapipe.eyes_open_detection import EyesOpenMediaPipe
from ml.headpose_estimation.headpose_estimation import HeadPoseEstimation
from ml.spoof_detection.anti_spoof_detection import AntiSpoofCheck
from ml import model_paths

face_detector = BlazeeFaceThales(weights_path=model_paths.BLAZEFAZE_BACK_WEIGHTS_PATH, anchors_path=model_paths.BLAZEFAZE_BACK_ANCHORS_PATH, debug=False, device='cpu')
head_pose_estimator = HeadPoseEstimation(model_path=model_paths.SYNERGYNET_MODEL_PATH, debug=False)
eye_open_detector = EyesOpenMediaPipe(model_path=model_paths.EYES_OPEN_MODEL_PATH)
anti_spoof_checker = AntiSpoofCheck(model_path=model_paths.SPOOF_MODEL_PATH)

def check_liveness(file_path: str) -> InternalLivenessResponse:
    """
    is_live
    msg
    """
    data_access = DataAccessImage(file_path=file_path)
    face_dicts, num_faces = face_detector.get_faces_batch(data_access_frames=[data_access], meta=None)
    print(f"face_dicts: {face_dicts}\nnum_faces: {num_faces}")


    if num_faces[0] > 1:
        return InternalLivenessResponse(is_live=False, msg=Messaging.MULTIPLE_FACES.value)
    
    if num_faces[0] == 0:
        return InternalLivenessResponse(is_live=False, msg=Messaging.NO_FACE.value)

    are_eyes_open = eye_open_detector.eyes_open_batch(data_layer_batch=[data_access], faces=face_dicts)
    print(f"are_eyes_open: {are_eyes_open}")

    if not are_eyes_open:
        return InternalLivenessResponse(is_live=False, msg=Messaging.EYES_OPEN.value)
    
    is_looking_straight = head_pose_estimator.get_head_pose(image_batch=[data_access], face_dicts=face_dicts)
    print(f"is_looking_straight: {is_looking_straight}")
    if not is_looking_straight:
        return InternalLivenessResponse(is_live=False, msg=Messaging.HEADPOSE.value)
    
    is_not_spoof = anti_spoof_checker.check_spoof(data_access_image=data_access)
    if not is_not_spoof:
        return InternalLivenessResponse(is_live=False, msg=Messaging.SPOOF.value)
    
    return InternalLivenessResponse(is_live=True, msg=Messaging.LIVE.value)


if __name__ == "__main__":
    file_path = '/Users/divyanshnew/Pictures/Photo on 21-12-22 at 11.23 AM.jpg'
    file_path = '/Users/divyanshnew/Pictures/Photo on 10-01-23 at 5.52 PM.jpg'
    response = check_liveness(file_path=file_path)
    print(f"Response: {asdict(response)}")