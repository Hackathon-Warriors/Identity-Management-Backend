from datetime import datetime, date
from uuid import UUID

from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

from flask.json import JSONEncoder
from flask import Flask

class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, UUID):
            return o.hex
        if isinstance(o, datetime):
            return o.timestamp()
        if isinstance(o, date):
            return float(o.strftime("%s"))
        return super().default(o)


class IAMFlaskApp(Flask):
    json_encoder = CustomJSONEncoder


def create_app():
    load_dotenv()
    appp = IAMFlaskApp(__name__)
    # Configure your app here
    initialize_blueprints(appp)
    return appp


def initialize_blueprints(appp):
    from .api.liveliness_checker.urls import bp as liveliness_bp
    from .api.urls import bp as health_check_bp
    appp.register_blueprint(health_check_bp)
    appp.register_blueprint(liveliness_bp)
