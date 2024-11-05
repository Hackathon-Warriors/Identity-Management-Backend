# core imports
import os
from functools import wraps
import logging
import uuid

# third party imports
from flask import jsonify, Response

from utils.logger import Logger

log = Logger(os.getenv('DEBUG_MODE'))


def to_json(status_code):
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            random_uuid = uuid.uuid4()
            log.info(f'Request recieved for id {random_uuid} with args--{args} and kwargs --{kwargs}')
            resp = func(*args, **kwargs)
            if not isinstance(resp, Response):
                resp = jsonify(resp)
            resp.status_code = status_code
            log.info(f'Response sent for {random_uuid} recieved with args--{resp.data}')
            return resp
        return inner
    return outer
