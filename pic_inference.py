# coding=utf-8
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

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

app = Flask(__name__)
model_id = None
classifier_clf = None


FLAGS = tf.app.flags.FLAGS

ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png'}

app._static_folder = FLAGS.static_folder
UPLOAD_FOLDER = FLAGS.static_folder + '/upload'
OUTPUT_FOLDER = FLAGS.static_folder + '/output'





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


def run_inference_on_image(image, model_file=None, label_file=None):
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
    create_graph(model_file)

    with tf.Session() as sess:
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

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        top_names = []
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            top_names.append(human_string)
            score = predictions[node_id]
            print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
    return predictions, top_k, top_names


def main(_):
    image = (FLAGS.image_file if FLAGS.image_file else
             os.path.join(FLAGS.model_dir, 'test.jpg'))
    run_inference_on_image(image)


def run_inference(modelfile, labelfile, image):
    pred,top_k,top_names = run_inference_on_image(image, modelfile, labelfile)
    return pred, top_k, top_names


@app.route('/')
def use_model():
    global classifier_clf
    global model_id
    classifier_clf = load_classifier(model_id)
    return render_template('/model/pic-inference.html')


@app.route('/model/picture-inference', methods=['GET', 'POST'])
def classifier_predict():
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print('file saved to %s' % file_path)
            return redirect(url_for('classifier_predict'))
    if request.method == 'GET':
        global image,model_file,label_file
        pred, top_k, top_names = run_inference_on_image(image, model_file, label_file)
        return str(top_k) + str(top_names)


if __name__ == '__main__':
    app.debug = False
    model_id = sys.argv[1]
    model_file = sys.argv[3]
    label_file = sys.argv[4]
    image_file = sys.argv[5]
    graph = tf.Graph()
    with graph.as_default():
        classify_graph_def = tf.GraphDef()
        print('classify_graph_def = tf.GraphDef()')
        with tf.gfile.GFile(model_file, 'rb') as f:
            classify_graph_def.ParseFromString(f.read())
            tf.import_graph_def(classify_graph_def, name='')
            print('tf.import_graph_def(classify_graph_def, name='')')
            classify_sess = tf.Session(graph=graph)
            print('classify_sess = tf.Session(graph=graph)')
    app.run(host='0.0.0.0', port=sys.argv[2], debug=False)
    # global model_id
    # model_id = 5
    # app.run(host='0.0.0.0', port="3444")