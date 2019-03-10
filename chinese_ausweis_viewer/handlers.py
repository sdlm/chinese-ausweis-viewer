import io

from flask import request, jsonify, send_file

from .utils.web_app import allowed_file
from .utils.predict_pipeline import predict_mask
from .app import app
from .exceptions import InvalidUsage


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
        # save_file(file)
        # return 'predict ...'

        input_file = file.read()

        mask = predict_mask(input_file)

        return send_file(
            io.BytesIO(mask),
            attachment_filename='mask.jpg',
            mimetype='image/jpg'
        )

    else:
        raise InvalidUsage('Only .jpg allowed')


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

