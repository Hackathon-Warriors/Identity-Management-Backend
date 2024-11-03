from flask.views import MethodView

from app.decorators import to_json

class HealthCheckAPI(MethodView):

    @to_json(200)
    def get(self):
        return {"message": "PONG PONG 12345678"}
