from flask import Flask, request, jsonify
from model import predict
from werkzeug.exceptions import BadRequest
import os


app = Flask(__name__)


@app.route("/prediction", methods=['POST'])
def prediction_api():
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")
    input_file.save('temp.mp3')
    genre = predict("temp.mp3")
    os.remove('temp.mp3')
    return jsonify({"genre": genre})
