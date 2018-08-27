# coding = utf-8
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
import os
import uuid
from os import path

app = Flask(__name__)
csv_path = 'csv/wine.csv'
classifier = None
classifier_clf = None

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png'}


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
        return jsonify("ok")


@app.route('/classfier-test', methods=['GET', 'POST'])
def classifier_test():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
        global classifier
        re = classifier.test_classifier(csv_path)
        print(re)
    return jsonify(re)


@app.route('/classifier-save', methods=['GET', 'POST'])
def classifier_save():
    if request.method == 'POST':
        name = request.form.get('name')
        global classifier
        classifier.save_clf(name)
        return jsonify(None)


# model
@app.route('/model/model-list.html')
def model_list():
    models = service.read_model_list()
    return render_template('/model/model-list.html', models=models)


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
        y = classifier_clf.predict(X).reshape(-1, 1)
        ans = np.hstack((y, X))
        ans = pd.DataFrame(ans)
        ans.to_csv('static/response.csv', index=False)
    return app.send_static_file("response.csv")


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


@app.route('/picture-classifier-upload', methods=['GET', 'POST'])
def upload_pic_file():
    if request.method == 'POST':
        print('test hello')
        file = request.files['file']
        print(file.filename)
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path, 'uploads/')
        file_name = upload_path + file.filename
        file.save(file_name)
    return redirect(url_for('picture_classifier'))


@app.route('/picture-classifier-params', methods=['GET', 'POST'])
def get_pic_train_params():
    if request.method == 'POST':
        params = {'learnRate': request.form.get('learnRateVal'), 'batchSize': request.form.get('batchSize'),
                  'checkPointPath': request.form.get('checkPointPath'), 'trainDir': request.form.get('trainDir'),
                  'excludeScopes': request.form.get('excludeScopes'), 'trainScopes': request.form.get('trainScopes'),
                  'optimizer': request.form.get('optimizer'), 'datasetName': request.form.get('datasetName'),
                  'modelName': request.form.get('modelName')}
        print(params)
    # base_path = path.abspath(path.dirname(__file__))
    with open('./uploads/path.txt', 'rb') as f:
        data_path = f.read().strip()
    cmd = """python ./slim/train_image_classifier.py --dataset_name=%s --dataset_dir=%s 
    --checkpoint_path=%s 
    --checkpoint_exclude_scopes=%s --trainable_scopes=%s
    --model_name=%s --train_dir=%s --learning_rate=%
    --optimizer=%s --batch_size=%s"""
    os.system(cmd % (
    params['datasetName'], data_path, params['checkPointPath'], params['excludeScopes'], params['trainScopes'],
    params['modelName'], params['trainDir'], params['learnRate'], params['optimizer'],
    params['batchSize']))
    return redirect(url_for('picture_classifier'))


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
