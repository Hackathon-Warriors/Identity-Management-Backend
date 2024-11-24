from app.models.asset import InternalLivenessResponse, Messaging


def check_liveness_v2(file_path: str) -> InternalLivenessResponse:
    return InternalLivenessResponse(is_live=True, msg=Messaging.LIVE.value)