import os
import traceback

from app import db
from app.constants.enums import LivelinessRequestStatus
from app.constants.error_constants import ErrorMessages
from app.db_models.liveliness import UserLivelinessData
from app.models.asset import InternalLivenessResponse
# from ml.liveliness_checker_v2 import check_liveness_v2
from ml.aggregate_liveness.liveness_checker import check_liveness

from utils.logger import Logger

from pathlib import Path
log = Logger()


class DocCheckerService:
    def __init__(self):
        pass

    @classmethod
    def verify_income_proof(cls, statement_doc, user_id: int, request_id: str, password: str):
        resp = dict(success=False, msg="", error_msg="")
        try:
            img_format = statement_doc.filename.split('.')[-1]
            if img_format not in ['jpg', 'jpeg', 'png','pdf']:
                resp['error_msg'] = ErrorMessages.INVALID_DOC_FORMAT.value
                return

            upload_folder = os.getenv('UPLOAD_FOLDER_LIVELINESS')
            if not os.path.exists(upload_folder):
                Path(upload_folder).mkdir(exists_ok=True, parents=True)

            save_filename = f'{request_id}_{statement_doc.filename}'
            save_path = os.path.join(upload_folder, save_filename)
            statement_doc.save(save_path)
            UserLivelinessData.insert_row(user_id, request_id, LivelinessRequestStatus.INITIATED.value, save_path)
            liveliness_resp: InternalLivenessResponse = check_liveness_v2(save_path)
            if liveliness_resp.is_live:
                UserLivelinessData.update_record_status(request_id, LivelinessRequestStatus.SUCCESS.value, None, True)
                resp['success'] = True
                resp['msg'] = "Liveness check passed"
            else:
                UserLivelinessData.update_record_status(request_id, LivelinessRequestStatus.FAILED.value, liveliness_resp.msg, False)
                resp['msg'] = liveliness_resp.msg
            db.session.commit()
            return resp
        except Exception as ex:
            db.session.rollback()
            log.error(f'Error while liveliness check for image {request_id}-- {ex}')
            return resp

    @classmethod
    def check_and_match_poi_face_data(cls, poi_doc, user_id: int, request_id: str, password: str):
        resp = dict(success=False, msg="", error_msg="")
        try:
            img_format = poi_doc.filename.split('.')[-1]
            if img_format not in ['jpg', 'jpeg', 'png', 'pdf']:
                resp['error_msg'] = ErrorMessages.INVALID_DOC_FORMAT.value
                return

            upload_folder = os.getenv('UPLOAD_FOLDER_LIVELINESS')
            if not os.path.exists(upload_folder):
                Path(upload_folder).mkdir(exists_ok=True, parents=True)

            save_filename = f'{request_id}_{poi_doc.filename}'
            save_path = os.path.join(upload_folder, save_filename)
            poi_doc.save(save_path)
            UserLivelinessData.insert_row(user_id, request_id, LivelinessRequestStatus.INITIATED.value, save_path)
            liveliness_resp: InternalLivenessResponse = check_liveness_v2(save_path)
            if liveliness_resp.is_live:
                UserLivelinessData.update_record_status(request_id, LivelinessRequestStatus.SUCCESS.value, None, True)
                resp['success'] = True
                resp['msg'] = "Liveness check passed"
            else:
                UserLivelinessData.update_record_status(request_id, LivelinessRequestStatus.FAILED.value, liveliness_resp.msg, False)
                resp['msg'] = liveliness_resp.msg
            db.session.commit()
            return resp
        except Exception as ex:
            db.session.rollback()
            log.error(f'Error while liveliness check for image {request_id}-- {ex}')
            return resp


