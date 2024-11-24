import os

from app.constants.error_constants import ErrorMessages
from app.models.asset import InternalLivenessResponse
from ml.aggregate_liveness.liveness_checker import check_liveness


class LivelinessCheckerService:
    def __init__(self):
        pass

    @classmethod
    def check_image_liveliness(cls, selfie_image):
        resp = dict(success=False, msg="", error_msg="")
        img_format = selfie_image.filename.split('.')[-1]
        if img_format not in ['jpg', 'jpeg', 'png']:
            resp['error_msg'] = ErrorMessages.INVALID_IMG_FORMAT.value
            return

        upload_folder = os.getenv('UPLOAD_FOLDER')
        if not os.path.exists(upload_folder):
            os.mkdir(upload_folder)

        save_path = os.path.join(upload_folder, selfie_image.filename)
        selfie_image.save(save_path)
        liveliness_resp: InternalLivenessResponse = check_liveness(save_path)
        if liveliness_resp.is_live:
            resp['success'] = True
            resp['msg'] = "Liveness check passed"
        else:
            resp['msg'] = liveliness_resp.msg
        return resp




