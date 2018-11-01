from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from Server.routes import index, image
app.register_blueprint(index.app, url_prefix='/')
app.register_blueprint(image.app, url_prefix='/image')
