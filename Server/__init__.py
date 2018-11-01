from flask import Flask
app = Flask(__name__)

from Server.routes import index, image
app.register_blueprint(index.app, url_prefix='/')
app.register_blueprint(image.app, url_prefix='/image')
