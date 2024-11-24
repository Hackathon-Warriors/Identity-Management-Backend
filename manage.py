from flask_script import Manager
from app import create_app

app = create_app()
manager = Manager(app)

@manager.command
def runserver():
    """Run the Flask development server."""
    app.run(debug=True, port=8001)

if __name__ == '__main__':
    manager.run()