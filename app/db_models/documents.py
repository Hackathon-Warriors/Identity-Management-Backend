from app import db
from app.db_models.base import Base


class UserDocumentData(Base):
    __tablename__ = 'user_documents_data'

    user_id = db.Column(db.BigInteger, nullable=False)
    request_id = db.Column(db.String(255))
    document_category = db.Column(db.String(255), nullable=True)
    document_format = db.Column(db.String(50), nullable=True)
    file_path = db.Column(db.String(255), nullable=True)

    @classmethod
    def update_or_insert_row(cls, user_id, request_id, document_category, document_format,file_path):
        existing_row = cls.query.filter(cls.user_id==user_id, cls.request_id==request_id, cls.document_category==document_category).first()
        if existing_row:
            existing_row.document_category = document_category
            existing_row.document_format = document_format
            existing_row.file_path = file_path
            db.session.commit()
            return existing_row

        return cls.insert(
            {'user_id': user_id, 'document_category': document_category, 'document_format': document_format,
             'request_id': request_id, 'file_path': file_path}
        )