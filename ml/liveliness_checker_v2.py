from app.models.asset import InternalLivenessResponse, Messaging, FaceMatchResponse


def check_liveness_v2(file_path: str) -> InternalLivenessResponse:
    return InternalLivenessResponse(is_live=True, msg=Messaging.LIVE.value)


def is_valid_pdf_v2(file_path: str)-> bool:
    return True


def match_faces_v2(selfie_path: str, doc_path: str):
    return FaceMatchResponse(is_similar=True,msg="")