from flask import Blueprint

from app.api.doc_checker.views import POIMatcherView, IncomeVerifierView

bp = Blueprint(
    'doc_checker', __name__, url_prefix='/api/<version>/docs'
)

bp.add_url_rule('/poi/match', view_func=POIMatcherView.as_view('check_and_match_poi'),
                methods=['POST'])

bp.add_url_rule('/statement/verify', view_func=IncomeVerifierView.as_view('verify_bank_statement'),
                methods=['POST'])
