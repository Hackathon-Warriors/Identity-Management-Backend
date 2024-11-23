from app.api.doc_checker import bp

bp.add_url_rule('/doc_checker', view_func=doc_checker, methods=['POST'])
