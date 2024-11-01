from flask import Blueprint
bp = Blueprint(
    'doc_checker', __name__, url_prefix='/api/<version>/liveliness_checker'
)