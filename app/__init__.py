import os
from datetime import datetime, date
from uuid import UUID

from dotenv import load_dotenv
from flask import Flask
from flask_redis import FlaskRedis
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.pool import QueuePool

db = SQLAlchemy()
redis_client = FlaskRedis()

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
    app = IAMFlaskApp(__name__)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'poolclass': QueuePool, 'max_overflow': 5, 'pool_size': 5}
    app.config['REDIS_URL'] = 'redis://127.0.0.1:6379/0'
    # Configure your app here
    initialize_blueprints(app)
    initialize_extensions(app)
    return app


def initialize_extensions(appp):
    """attach all the extensions to the application instance"""
    db.init_app(appp)
    redis_client.init_app(appp)


def initialize_blueprints(appp):
    from .api.liveliness_checker.urls import bp as liveliness_bp
    from .api.urls import bp as health_check_bp
    appp.register_blueprint(health_check_bp)
    appp.register_blueprint(liveliness_bp)

# enable app to discover all models
from app import db_models  # NOQA