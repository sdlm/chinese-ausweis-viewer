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
    file = get_file_from_request(request)

    input_file = file.read()

    mask = predict_mask(input_file)

    return send_file(
        io.BytesIO(mask),
        attachment_filename='mask.jpg',
        mimetype='image/jpg'
    )


@app.route('/pipeline/', methods=['POST'])
def pipeline():
    file = get_file_from_request(request)

    input_file = file.read()

    card_fields_data = extract_card_fields_data(input_file)

    return jsonify(card_fields_data)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def get_file_from_request(request, rise_exc: bool = True):
    if 'file' not in request.files:
        if rise_exc:
            raise InvalidUsage('You must send file')
        return None

    file = request.files['file']
    if not file or file.filename == '':
        if rise_exc:
            raise InvalidUsage('You must send file')
        return None

    if not allowed_file(file.filename):
        if rise_exc:
            raise InvalidUsage('Only .jpg allowed')
        return None

    return file
