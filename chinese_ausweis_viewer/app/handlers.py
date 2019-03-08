from flask import request, jsonify

from .app import app
from .exceptions import InvalidUsage
from .utils import allowed_file, save_file


@app.route('/')
def ping():
    return {'status': 'ok'}


@app.route('/predict/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        raise InvalidUsage('You must send file')
    file = request.files['file']
    if not file or file.filename == '':
        raise InvalidUsage('You must send file')
    if allowed_file(file.filename):
        save_file(file)
        return 'predict ...'
    else:
        raise InvalidUsage('Only .jpg allowed')


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

