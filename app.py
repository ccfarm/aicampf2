# -*- coding:UTF-8 -*-
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
from flask import redirect
import json
import uuid, os
import tensorflow as tf
from classifier import Classifier
import pic_classifier as pc
import csv
from os import path

app = Flask(__name__)
csv_path = 'csv/wine.csv'
classifier = None

# tf.app.flags.DEFINE_string('slim_checkpoint_dir', '../slim/checkpoint', '')
# tf.app.flags.DEFINE_string('dataset_dir', './data', '')
# tf.app.flags.DEFINE_string('static_folder', './static', '')
# tf.app.flags.DEFINE_string('output_dir', './output', '')
# tf.app.flags.DEFINE_integer('num_top_predictions', 5,
#                             """Display this many predictions.""")
#
# FLAGS = tf.app.flags.FLAGS

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png'}


# UPLOAD_FOLDER = FLAGS.static_folder + '/upload'
# OUTPUT_FOLDER = FLAGS.static_folder + '/output'


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


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


@app.route('/picture-classifier-upload', methods=['GET', 'POST'])
def upload_pic_file():
    if request.method == 'POST':
        print('test hello')
        file = request.files['file']
        # old_file_name = file.filename
        # if file and allowed_files(old_file_name):
        # filename = rename_filename(old_file_name)
        # file_path = os.path.join(UPLOAD_FOLDER, filename)
        # file.save(file_path)
        # 这段逻辑到时候按需修正 先把demo跑通
        print(file.filename)
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path, 'uploads/')
        file_name = upload_path + file.filename
        file.save(file_name)
        print(file_name)
    return redirect(url_for('picture_classifier'))


def trian(params):
    command = ""
    os.system(command)


@app.route('/picture-classifier-params', methods=['GET', 'POST'])
def get_pic_train_params():
    if request.method == 'POST':
        params = {'learnRate': request.form.get('learnRateVal'), 'batchSzie': request.form.get('batchSzie'),
                  'checkPointPath': request.form.get('checkPointPath'), 'trainDir': request.form.get('trainDir'),
                  'excludeScopes': request.form.get('excludeScopes'), 'trainScopes': request.form.get('trainScopes'),
                  'optimizer': request.form.get('optimizer'), 'datasetName': request.form.get('datasetName'),
                  'modelName': request.form.get('modelName')}
        print(params)
    # base_path = path.abspath(path.dirname(__file__))
    with open('./uploads/file.txt', 'rb') as f:
        data_path = f.read().strip()
    cmd = """python train_image_classifier.py --dataset_name=cifar10 --dataset_dir=./slim/tmp/cifar10 
    --checkpoint_path=./slim/tmp/pnasnet-5_large_2017_12_13/model.ckpt 
    --checkpoint_exclude_scopes=final_layer,aux_7 --trainable_scopes=final_layer,aux_7,cell10,cell11 
    --model_name=pnasnet_large --train_dir=./slim/tmp/pnasnet-model --learning_rate=0.001 
    --optimizer=rmsprop --batch_size=8 --clone_on_cpu=True"""
    os.system(cmd)
    return redirect(url_for('picture_classifier'))


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
        # f = request.files['the_file']
        # data = json.loads(request.form.get('data'))
        # f.save("/csv/" + f.name)
        # reader = csv.reader(f)
        global csv_path
        global classifier
        classifier = Classifier(csv_path)
        score = classifier.get_score()
        re = {"score": score}
        return jsonify(re)


@app.route('/classifier-save', methods=['GET', 'POST'])
def classifier_save():
    if request.method == 'POST':
        name = request.form.get('name')
        global classifier
        classifier.save_clf(name)
        return jsonify(None)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
