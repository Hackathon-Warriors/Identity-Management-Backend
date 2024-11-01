from uuid import UUID

from sqlalchemy import text

from app import db


class Base(db.Model):
    """
    All db_models will inherit from this base model
    """
    __abstract__ = True
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(
        db.DateTime, default=db.func.now(), onupdate=db.func.now()
    )

    @classmethod
    def raw_query(cls, sql_query):
        with db.engine.connect() as connection:
            result = connection.execute(text(sql_query))
            return result

    @classmethod
    def insert(cls, row):
        row = cls(**row)
        db.session.add(row)
        db.session.commit()
        return row

    @classmethod
    def fields(cls):
        for column in cls.__table__.columns:
            yield column.name

    def to_dict(self):
        d = {}
        for column in self.fields():
            value = getattr(self, column)
            if isinstance(value, UUID):
                d[column] = str(value)
                continue
            d[column] = value
        return d

    @classmethod
    def all(cls, query=None, page=None, limit=10):
        query = cls.query if not query else query

        if page:
            offset = (page - 1) * limit
            query = query.offset(offset).limit(limit)
        return query
