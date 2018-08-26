from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
from flask import redirect
import json
from classifier import Classifier
import csv

app = Flask(__name__)
csv_path = ''

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

@app.route('/classfier-upload', methods=['GET', 'POST'])
def upload_classifier_file():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
    return redirect(url_for('classifier'))

@app.route('/classifier-train', methods=['GET', 'POST'])
def classifier_train():
    if request.method == 'POST':
        #f = request.files['the_file']
        #data = json.loads(request.form.get('data'))
        # f.save("/csv/" + f.name)
        # reader = csv.reader(f)
        global csv_path
        classifier = Classifier(csv_path)
        score = classifier.get_score()
        re = {"score":score}
        return jsonify(re)






if __name__ == '__main__':
    app.run()
