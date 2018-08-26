from flask import Flask
from flask import render_template
from flask import request
import csv

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index/index.html')

@app.route('/welcome.html')
def welcome():
    return render_template('index/welcome.html')

# train
@app.route('/train/picture-classifier.html')
def picture_classifier():
    return render_template('train/picture-classifier.html')

@app.route('/train/classifier.html')
def classifier():
    return render_template('train/classifier.html')

@app.route('/train/classifier/upload', methods=['GET', 'POST'])
def classifier_upload():
    if request.method == 'POST':
        f = request.files['the_file']
        reader = csv.reader(f)


if __name__ == '__main__':
    app.debug=True
    app.run()
