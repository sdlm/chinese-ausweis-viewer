from flask import Flask

from . import settings

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
app.secret_key = 'super secret key'

from . import handlers
