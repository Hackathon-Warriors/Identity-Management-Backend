from app import db
from app.db_models.base import Base


class UserLivelinessData(Base):
    __tablename__ = 'user_liveliness_data'

    user_id = db.Column(db.BigInteger, nullable=False)
    checks_passed = db.Column(db.Boolean, default=False)
    request_id = db.Column(db.String(255))
    req_status = db.Column(db.String(50))
    error_msg = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(255), nullable=True)

    @classmethod
    def insert_row(cls, user_id, request_id, req_status, file_path):
        cls.insert(
            {'user_id': user_id, 'request_id': request_id, 'req_status': req_status, 'file_path': file_path}
        )

    @classmethod
    def get_latest_live_record_by_userid(cls, user_id: int):
        return cls.query.filter(cls.user_id == user_id, cls.checks_passed.is_(True)).order_by(cls.id.desc()).first()

    @classmethod
    def update_record_status(cls, request_id, req_status, error_msg, checks_passed):
        update_meta = dict(req_status=req_status, error_msg=error_msg, checks_passed=checks_passed)
        cls.query.filter(cls.request_id == request_id).update(update_meta)
