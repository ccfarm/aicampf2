from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from subprocess import Popen, PIPE, STDOUT
import sys
import mysql.connector
import const

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

print(sys.getdefaultencoding())

app = Flask(__name__)
model_id = None
model_file_path = None
inference_file_path = None


FLAGS = tf.app.flags.FLAGS

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png'}


UPLOAD_FOLDER = 'uploads'
# OUTPUT_FOLDER = FLAGS.static_folder + '/output'



class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)

    def load(self, label_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human

        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]



def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


def create_graph(model_file=None):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    if not model_file:
        model_file = FLAGS.model_file
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, sess=None):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = open(image, 'rb').read()

    # Creates graph from saved GraphDef.
    # create_graph(model_file)

    # with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('output:0')
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(label_file)

    top_k = predictions.argsort()[-5:][::-1]
    top_names = []
    scores = []
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        top_names.append(human_string)
        score = predictions[node_id]
        scores.append(score)
        print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
    return predictions, scores, top_names


def main(_):
    image = (FLAGS.image_file if FLAGS.image_file else
             os.path.join(FLAGS.model_dir, 'test.jpg'))
    run_inference_on_image(image)


def run_inference(modelfile, labelfile, image):
    pred,top_k,top_names = run_inference_on_image(image, modelfile, labelfile)
    return pred, top_k, top_names

def get_model_path(id):
    conn = mysql.connector.connect(user=const.db_user_name, password=const.db_password, database=const.db_database
                                   , auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    cursor.execute('select * from model_manage where id = %s', [id])
    model = cursor.fetchone()
    cursor.close()
    conn.close()
    path = model[4]
    return path

@app.route('/')
def use_model():
    global model_file_path
    global model_id
    model_file_path = get_model_path(model_id)
    return render_template('/model/pic-inference.html')


@app.route('/model/picture-inference', methods=['GET', 'POST'])
def classifier_predict():
    print('classfier_predict')
    result = """
        <!doctype html>
        <title>Picture classify</title>
        <p>%s</p>
        """ % "<br>"
    if request.method == 'POST':
        print('ENTER POST')
        global inference_file_path, model_file_path, label_file, port
        f = request.files['file']
        # base_path = path.abspath(path.dirname(__file__))
        # upload_path = path.join('/static/upload/', f.filename)
        inference_file_path = 'static/upload/' + f.filename
        f.save(inference_file_path)
        print(f.filename)
        pred, scores, top_names = run_inference_on_image(inference_file_path, sess)

        new_url = 'http://39.104.63.247:' + str(port) + '/' + inference_file_path
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url

        format_string = '<b><h3>the result is as below:<h3></b><br/>'

        for i in range(5):
            format_string += "%s (similarity:%.5f&#37;)<BR>" % (top_names[i], scores[i] * 100)
            format_string += '\n'
        return new_tag + format_string + '<BR>'
    return redirect(url_for('classifier_predict'))


if __name__ == '__main__':
    app.debug = False
    model_id = sys.argv[1]
    model_file = sys.argv[3]
    label_file = sys.argv[4]
    # image_file = sys.argv[5]
    port = sys.argv[2]
    graph = tf.Graph()
    with graph.as_default():
        classify_graph_def = tf.GraphDef()
        print('classify_graph_def = tf.GraphDef()')
        with tf.gfile.GFile(model_file, 'rb') as f:
            # with open(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            sess = tf.Session(graph=graph)

    app.run(host='0.0.0.0', port=port, debug=False)
    # global model_id
    # model_id = 5
    # app.run(host='0.0.0.0', port="3444")
    run_inference_on_image('./slim/test.jpg', sess)
