from app.api.liveliness_checker.views import ImageLivelinessCheckerView

from flask import Blueprint
bp = Blueprint(
    'liveliness_checker', __name__, url_prefix='/api/<version>/liveliness'
)

bp.add_url_rule('/check', view_func=ImageLivelinessCheckerView.as_view('check_image_livelines'), methods=['POST'])