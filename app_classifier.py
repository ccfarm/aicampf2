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

from regression import Regression
from regression import load_regression
import csv
import os
import uuid
from os import path
from subprocess import Popen, PIPE, STDOUT
import sys

app = Flask(__name__)
model_id = None
classifier_clf = None

@app.route('/')
def use_model():
    global classifier_clf
    global model_id
    classifier_clf = load_classifier(model_id)
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

if __name__ == '__main__':
    app.debug = False
    model_id = sys.argv[1]
    app.run(host='0.0.0.0', port=sys.argv[2])
    # global model_id
    # model_id = 5
    # app.run(host='0.0.0.0', port="3444")