from flask import request, jsonify
from flask.views import MethodView

from app.services.doc_check_service import DocCheckerService
from utils.logger import Logger

log = Logger()


class POIMatcherView(MethodView):

    @classmethod
    def post(cls, version):
        try:
            if 'poi_doc' not in request.files:
                raise Exception('poi document not found in request')
            poi_doc = request.files['poi_doc']
            request_id = request.form['request_id']
            user_id = request.form['user_id']
            doc_password = request.form['password']
            log.info(f"POIMatcherView :: poi document recieved :: {poi_doc}")
            resp = DocCheckerService.check_and_match_poi_face_data(poi_doc, user_id, request_id, doc_password)
            if resp.get('success'):
                return jsonify(resp), 200
            else:
                return jsonify(resp), 400
        except Exception as e:
            log.error(f'POIMatcherView error :: {e}')
            return jsonify({"success": False, "error_msg": f"error while reading document{e}"}), 400


class IncomeVerifierView(MethodView):

    @classmethod
    def post(cls, version):
        try:
            if 'statement_doc' not in request.files:
                raise Exception('statement not found in request')
            poi_doc = request.files['statement_doc']
            request_id = request.form['request_id']
            user_id = request.form['user_id']
            doc_password = request.form['password']
            log.info(f"IncomeVerifierView :: statement recieved :: {poi_doc}")
            resp = DocCheckerService.verify_income_proof(poi_doc, user_id, request_id, doc_password)
            if resp.get('success'):
                return jsonify(resp), 200
            else:
                return jsonify(resp), 400
        except Exception as e:
            log.error(f'IncomeVerifierView error :: {e}')
            return jsonify({"success": False, "error_msg": f"error while reading document{e}"}), 400
