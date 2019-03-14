import io

from flask import request, jsonify, send_file

from .utils.web_app import allowed_file
from .utils.predict_pipeline import predict_mask, extract_card_fields_data, prepare_output_img, \
    get_np_arr_from_bytes
from .app import app
from .exceptions import InvalidUsage


@app.route('/')
def ping():
    return {'status': 'ok'}


@app.route('/predict/', methods=['POST'])
def predict():
    file = get_file_from_request(request)
    img_bytes = file.read()

    img_arr = get_np_arr_from_bytes(img_bytes)
    mask_arr = predict_mask(img_arr)
    mask_image_file = prepare_output_img(mask_arr)

    return send_file(
        io.BytesIO(mask_image_file),
        attachment_filename='mask.jpg',
        mimetype='image/jpg'
    )


@app.route('/pipeline/', methods=['POST'])
def pipeline():
    file = get_file_from_request(request)
    img_bytes = file.read()

    card_fields_data = extract_card_fields_data(img_bytes)

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
    print('- ' * 10)
    print('file.filename: %s' % file.filename)
    print('file.content_length: %s' % file.content_length)
    print('file.mimetype: %s' % file.mimetype)
    print('file.mimetype_params: %s' % file.mimetype_params)
    print('- ' * 10)
    if not file or file.filename == '':
        if rise_exc:
            raise InvalidUsage('You must send file')
        return None

    if not allowed_file(file.filename):
        if rise_exc:
            raise InvalidUsage('Only .jpg allowed')
        return None

    # if file.content_length == 0:
    #     if rise_exc:
    #         raise InvalidUsage('Empty file')
    #     return None

    return file
