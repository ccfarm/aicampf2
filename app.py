from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
from flask import redirect
from flask import app
import service
import pandas as pd
import numpy as np
import json
from classifier import Classifier
from classifier import load_classifier
import csv

app = Flask(__name__)
csv_path = 'csv/wine.csv'
classifier = None
classifier_clf = None

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

# @app.route('/classfier-upload', methods=['GET', 'POST'])
# def upload_classifier_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         global csv_path
#         csv_path = "csv/" + f.filename
#         f.save(csv_path)
#     return redirect(url_for('classifier'))

@app.route('/classfier-upload', methods=['GET', 'POST'])
def upload_classifier_file():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
    return jsonify("ok")

@app.route('/classifier-train', methods=['GET', 'POST'])
def classifier_train():
    if request.method == 'POST':
        global csv_path
        global classifier
        param = request.form.get('param')
        param_dict = {}
        param_list = param.split('&')
        for p in param_list:
            tmp = p.split('=')
            if tmp[0] == 'tol':
                tmp[1] = float(tmp[1])
            elif tmp[0] == 'C':
                tmp[1] = float(tmp[1])
            elif tmp[0] == 'max_iter':
                tmp[1] = int(tmp[1])
            param_dict[tmp[0]] = tmp[1]

        classifier = Classifier(csv_path, param_dict)
        score = classifier.get_score()
        re = {"score":score}
        return jsonify(re)

@app.route('/classifier-save', methods=['GET', 'POST'])
def classifier_save():
    if request.method == 'POST':
        name = request.form.get('name')
        global classifier
        classifier.save_clf(name)
        return jsonify(None)

#model
@app.route('/model/model-list.html')
def model_list():
    models = service.read_model_list()
    return render_template('/model/model-list.html', models = models )

@app.route('/model/use-model.html/<int:id>')
def use_model(id):
    global classifier_clf
    classifier_clf = load_classifier(id)
    return render_template('/model/use-model.html')

@app.route('/model/classfier-predict', methods=['GET', 'POST'])
def classifier_predict():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
        X = pd.read_csv(csv_path)
        global classifier_clf
        y = classifier_clf.predict(X).reshape(-1,1)
        ans = np.hstack((y, X))
        ans = pd.DataFrame(ans)
        filepath = 'static/response.csv'
        ans.to_csv(filepath, index=False)
    return app.send_static_file("response.csv")

if __name__ == '__main__':
    app.debug=True
    app.run()
