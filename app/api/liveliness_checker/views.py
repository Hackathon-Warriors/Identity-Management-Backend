from flask import request, jsonify
from flask.views import MethodView

from app.services.liveliness_service import LivelinessCheckerService
from utils.logger import Logger

log = Logger()


class ImageLivelinessCheckerView(MethodView):

    @classmethod
    def post(cls, version):
        try:
            if 'selfie_image' not in request.files:
                raise Exception('selfie_image not found in request')
            selfie_image = request.files['selfie_image']
            log.info(f"ImageLivelinessCheckerView :: selfie_image recieved :: {selfie_image}")
            resp = LivelinessCheckerService.check_image_liveliness(selfie_image)
            if resp.get('success'):
                return jsonify(resp), 200
            else:
                return jsonify(resp), 400
        except Exception as e:
            log.error(f'ImageLivelinessCheckerView error :: {e}')
            return jsonify({"success": False, "error_msg": f"error while reading image{e}"}), 400
