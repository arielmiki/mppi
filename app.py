from flask import Flask, request, jsonify, render_template
from model import predict
from werkzeug.exceptions import BadRequest
import os


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def prediction_api():
    if request.method == 'GET':
        return render_template('index.html')
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")
    input_file.save('temp.mp3')
    genre = predict("temp.mp3")
    os.remove('temp.mp3')
    return render_template('index.html', genre=genre)

if __name__ == '__main__':
    app.run(debug=True)