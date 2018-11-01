from flask import Blueprint

app = Blueprint('index', __name__)

@app.route('/', methods=['GET'])
def index_main():
    return "<h1>Index Route</h1>"

