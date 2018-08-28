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
import const as const
from regression import Regression
from regression import load_regression
import csv
import os, json
import uuid
from os import path
from subprocess import Popen, PIPE, STDOUT
import mysql.connector

app = Flask(__name__)
csv_path = 'csv/wine.csv'
classifier = None
classifier_clf = None

regression = None
regression_clf = None

model_params = {}

size = 0
offset = 0
p = None

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png'}

model_online_map = {}
model_popen_map = {}


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


@app.route('/train/regression.html')
def regression():
    return render_template('train/regression.html')


@app.route('/classfier-upload', methods=['GET', 'POST'])
def upload_classifier_file():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
    return jsonify("ok")


@app.route('/classifier-show-data', methods=['GET', 'POST'])
def classifier_show_file():
    global csv_path
    f = open(csv_path)
    content = ''
    reader = csv.reader(f)
    rows = [row for row in reader]
    f.close()
    return render_template('train/classifier-show-data.html', rows=rows)


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
        re = {"score": score}
        return jsonify(re)



@app.route('/classifier-test', methods=['GET', 'POST'])
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


@app.route('/regression-upload', methods=['GET', 'POST'])
def upload_regression_file():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
    return jsonify("ok")


@app.route('/regression-train', methods=['GET', 'POST'])
def regression_train():
    if request.method == 'POST':
        global csv_path
        global regression
        param = request.form.get('param')
        param_dict = {}
        param_list = param.split('&')
        for p in param_list:
            tmp = p.split('=')
            if tmp[0] == 'n_estimators':
                tmp[1] = int(tmp[1])
            elif tmp[0] == 'max_features ':
                tmp[1] = str(tmp[1])
            elif tmp[0] == 'max_depth':
                tmp[1] = int(tmp[1])
            param_dict[tmp[0]] = tmp[1]

        regression = Regression(csv_path, param_dict)
        score = regression.get_score()
        re = {"score": score}
        return jsonify(re)


@app.route('/regression-test', methods=['GET', 'POST'])
def regression_test():
    if request.method == 'POST':
        f = request.files['file']
        global csv_path
        csv_path = "csv/" + f.filename
        f.save(csv_path)
        global regression
        re = regression.test_regression(csv_path)
        print(re)
    return jsonify(re)


@app.route('/regression-save', methods=['GET', 'POST'])
def regression_save():
    if request.method == 'POST':
        name = request.form.get('name')
        global regression
        regression.save_clf(name)
        return jsonify(None)


# model


# model

@app.route('/model/model-list.html')
def model_list():
    models = service.read_model_list()
    global model_online_map
    return render_template('/model/model-list.html', models=models, model_online_map=model_online_map)


@app.route('/model/use-model.html/<int:id>')
def use_model(id):
    global classifier_clf
    classifier_clf = load_classifier(id)
    return render_template('/model/use-model.html')

def get_model_effect(id):
    conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                   , auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    cursor.execute('select * from model_manage where id = %s', [id])
    model = cursor.fetchone()
    cursor.close()
    conn.close()
    effect = model[3]
    model_path = model[4]
    return effect, model_path


@app.route('/model/start-model/<int:id>')
def start_model(id):
    port = "67" + str(id)
    url = "http://39.104.63.247:" + port
    global model_online_map
    global model_popen_map
    model_online_map[id] = url
    effect, model_path = get_model_effect(id)
    if effect == 'pic_classification':
        cmd = """python pic_inference.py %s %s %s ./slim/data/labels.txt"""
        p = Popen(cmd % (str(id), port, model_path), shell=True, stdout=PIPE)
    else:
        p = Popen("python app_classifier.py " + str(id) + " " + port, shell=True, stdout=PIPE)
    model_popen_map[id] = p
    return redirect(url_for('model_list'))

@app.route('/model/shut-model/<int:id>')
def shut_model(id):
    global model_online_map
    global model_popen_map
    model_online_map.pop(id)
    p = model_popen_map[id]
    p.kill()
    model_popen_map.pop(id)
    return redirect(url_for('model_list'))


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

@app.route('/pict-classifier-getLog', methods=['GET', 'POST'])
def get_pic_train_log():
    global offset
    if request.method == 'GET':
        with open('log.txt', 'r') as f:
            f.seek(offset, 0)
            f.seek(0, 2)
            cur_length = f.tell()
            size = cur_length - offset
            f.seek(offset, 0)
            data = f.read(size)
            offset = cur_length
            return data


@app.route('/pict-classifier-getEvalLog', methods=['GET', 'POST'])
def get_pic_eval_log():
    global offset
    if request.method == 'GET':
        with open('logEval.txt', 'r') as f:
            f.seek(offset, 0)
            f.seek(0, 2)
            cur_length = f.tell()
            size = cur_length - offset
            f.seek(offset, 0)
            data = f.read(size)
            offset = cur_length
            return data

@app.route('/picture-classifier-params', methods=['GET', 'POST'])
def get_pic_train_params():
    global p, model_params
    if request.method == 'POST':
        params = {'learnRate': request.form.get('learnRateVal'), 'batchSize': request.form.get('batchSize'),
                  'checkPointPath': request.form.get('checkPointPath'), 'trainDir': request.form.get('trainDir'),
                  'excludeScopes': request.form.get('excludeScopes'), 'trainScopes': request.form.get('trainScopes'),
                  'optimizer': request.form.get('optimizer'), 'datasetName': request.form.get('datasetName'),
                  'modelName': request.form.get('modelName')}
        print(params)
        model_params = params
        # base_path = path.abspath(path.dirname(__file__))
        fo = open('log.txt','w')
        with open('./uploads/path.txt', 'rb') as f:
            data_path = f.read().strip()
        cmd = """python -u ./slim/train_image_classifier.py --dataset_name=%s --dataset_dir=%s \
        --checkpoint_path=%s --checkpoint_exclude_scopes=%s --trainable_scopes=%s \
        --model_name=%s --train_dir=%s --learning_rate=%s \
        --optimizer=%s --batch_size=%s"""
        p = Popen(cmd % (
        params['datasetName'], data_path, params['checkPointPath'], params['excludeScopes'], params['trainScopes'],
        params['modelName'], params['trainDir'], params['learnRate'], params['optimizer'],
        params['batchSize']), shell=True, stdin=PIPE, stdout=fo, stderr=STDOUT, close_fds=True)
        # for line in p.stdout:
        #     print(line)
    global size, offset
    if request.method == 'GET' and p is not None:
        print('enter get method')
        signal = request.args.get('signal')
        print(signal)
        # if signal == 'LOG':
        #     # p.stdout.seek(0, 2)
        #     # size += p.stdout.tell()
        #     data = p.stdout.read()
        #     print(data)
        #     # offset = size
        #     return data
        if signal == 'STOP':
            p.kill()
            return 'Model stop train'
    return redirect(url_for('picture_classifier'))


@app.route('/picture-eval', methods=['GET', 'POST'])
def get_pic_eval_params():
    print('eval_function')
    if request.method == 'POST':
        params = {'datasetName_eval': request.form.get('datasetName_eval'),
                  'batchSize_eval': request.form.get('batchSize_eval'),
                  'maxNumBatches': request.form.get('maxNumBatches')}
        print(params)
        fo = open('logEval.txt','w')
        cmd = """python ./slim/eval_image_classifier.py --dataset_name=%s --dataset_dir=./slim/tmp/cifar10 \
        --dataset_split_name=test --model_name=pnasnet_large --checkpoint_path=./slim/tmp/pnasnet-model \
        --eval_dir=./slim/tmp/pnasnet-model --batch_size=%s --max_num_batches=%s"""
        # os.system(cmd % (params['datasetName_eval'], params['batchSize_eval'], params['maxNumBatches']))

        p = Popen(cmd % (params['datasetName_eval'], params['batchSize_eval'], params['maxNumBatches']), shell=True, stdin=PIPE, stdout=fo, stderr=STDOUT, close_fds=True)
        return redirect(url_for('picture_classifier'))



@app.route('/picture-model-export', methods=['GET', 'POST'])
def get_pic_export():
    if request.method == 'POST':
        params = {'modelSaveName': request.form.get('modelSaveName')}
        print(params)
        base_path = path.abspath(path.dirname(__file__))
        cmd1 = """python ./slim/export_inference_graph_new.py --model_name=pnasnet_large --batch_size=1 \
        --dataset_name=cifar10 --dataset_dir=./tmp/cifar10 --output_file=pnasnet_graph_def.pb"""
        os.system(cmd1)
        all_files = os.listdir(r'./slim/tmp/pnasnet-model')
        maxnum = '0'
        maxfile = None
        for file in all_files:
            if file.count('index'):
                num = file.split('.')[-2].split('-')[-1]
                if num > maxnum:
                    maxnum = num
                    maxfile = file
        maxfile = maxfile.split('.')[0] + '.' + maxfile.split('.')[1]
        cmd2 = """python ./slim/freeze_graph.py --input_graph=pnasnet_graph_def.pb \
        --input_checkpoint=./slim/tmp/pnasnet-model/%s --output_graph=%s \
         --output_node_names=output --input_binary=True"""
        model_save_path = "model/" + params['modelSaveName']
        print(model_save_path)
        os.system(cmd2 % (maxfile, model_save_path))
        save_clf(model_save_path, params['modelSaveName'])
        return redirect(url_for('picture_classifier'))


def save_clf(path, name):
    global model_params
    model_path = path
    params_str = json.dumps(model_params)
    conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                   ,auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    cursor.execute('insert into model_manage (model_name, model_zhonglei, model_zuoyong, model_address, model_config)'
                   'values (%s, %s, %s, %s, %s)', [name, 'pnasnet', 'pic_classification', model_path, params_str])
    conn.commit()
    cursor.close()
    conn.close()
    return None


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',debug=False)
