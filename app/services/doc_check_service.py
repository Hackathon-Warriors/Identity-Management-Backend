import os
import traceback

from app import db
from app.constants.enums import LivelinessRequestStatus, DocumentTypes
from app.constants.error_constants import ErrorMessages
from app.db_models.liveliness import UserLivelinessData
from app.db_models.documents import UserDocumentData
from app.models.asset import FaceMatchResponse
from ml.pdf_validation.pdf_validator import is_valid_pdf

from utils.logger import Logger

from pathlib import Path
log = Logger()


class DocCheckerService:
    def __init__(self):
        pass

    @classmethod
    def verify_income_proof(cls, statement_doc, user_id: int, request_id: str, password: str):
        resp = dict(success=False, error_msg="", data=dict(is_doc_valid=False))
        try:
            doc_format = statement_doc.filename.split('.')[-1]
            if doc_format not in ['pdf']:
                resp['error_msg'] = ErrorMessages.INVALID_DOC_FORMAT.value
                return

            upload_folder = os.getenv('UPLOAD_FOLDER_INCOME')
            if not os.path.exists(upload_folder):
                Path(upload_folder).mkdir(exist_ok=True, parents=True)

            existing_livliness_record = UserLivelinessData.get_latest_live_record_by_userid(user_id)
            if not existing_livliness_record:
                resp['error_msg'] = ErrorMessages.SELFIE_STEP_INCOMPLETE.value
                return resp

            save_filename = f'{request_id}_{statement_doc.filename}'
            save_path = os.path.join(upload_folder, save_filename)
            statement_doc.save(save_path)
            UserDocumentData.update_or_insert_row(user_id, request_id, DocumentTypes.BANK_STATEMENT.value, doc_format, save_path)
            is_doc_valid: bool = is_valid_pdf(save_path)
            if is_doc_valid:
                resp['success'] = True
                resp['data']['is_doc_valid'] = True
            else:
                resp['success'] = True
                resp['error_msg'] = "Invalid income proof uploaded, Please upload correct document"
            return resp
        except Exception as ex:
            db.session.rollback()
            log.error(f'Error while income check for document {request_id}-- {ex}')
            return resp

    @classmethod
    def check_and_match_poi_face_data(cls, poi_doc, user_id: int, request_id: str, password: str):
        resp = dict(success=False, error_msg="", data=dict(is_similar=False))
        try:
            doc_format = poi_doc.filename.split('.')[-1]
            if doc_format not in ['jpg', 'jpeg', 'png', 'pdf']:
                resp['error_msg'] = ErrorMessages.INVALID_DOC_FORMAT.value
                return

            upload_folder = os.getenv('UPLOAD_FOLDER_FACEMATCH')
            if not os.path.exists(upload_folder):
                Path(upload_folder).mkdir(exist_ok=True, parents=True)

            save_filename = f'{request_id}_{poi_doc.filename}'
            save_path = os.path.join(upload_folder, save_filename)
            poi_doc.save(save_path)
            existing_livliness_record = UserLivelinessData.get_latest_live_record_by_userid(user_id)
            if not existing_livliness_record:
                resp['error_msg'] = ErrorMessages.SELFIE_STEP_INCOMPLETE.value
                return resp

            UserDocumentData.update_or_insert_row(user_id, request_id, DocumentTypes.POI.value, doc_format, save_path)
            face_match_resp: FaceMatchResponse = match_faces(existing_livliness_record.file_path, save_path)
            if face_match_resp.is_similar:
                UserLivelinessData.update_record_status(request_id, LivelinessRequestStatus.SUCCESS.value, None, True)
                resp['success'] = True
                resp['data']['is_similar'] = True
            else:
                UserLivelinessData.update_record_status(request_id, LivelinessRequestStatus.FAILED.value, face_match_resp.msg, False)
                resp['msg'] = face_match_resp.msg
            db.session.commit()
            return resp
        except Exception as ex:
            db.session.rollback()
            log.error(f'Error while face match check for image {request_id}-- {ex}')
            return resp


