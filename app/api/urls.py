from .views import HealthCheckAPI
from flask import Blueprint

bp = Blueprint(
    'app-health', __name__, url_prefix='/knock-knock'
)
bp.add_url_rule('/check', view_func=HealthCheckAPI.as_view('health_check'), methods=['GET'])