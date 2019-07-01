import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
import traceback

import numpy as np
import uvloop

from classify_nsfw import caffe_preprocess_and_compute, load_model

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin

try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)
def classify(image_path: str) -> np.float64:
    with open(image_path, "rb") as image:
        scores = caffe_preprocess_and_compute(image.read(), caffe_transformer=caffe_transformer, caffe_net=nsfw_net, output_layers=["prob"])
    return scores[1]



@app.route("/detect", methods=["POST"])
def detect():

    input_path = generate_random_filename(upload_directory,"jpg")

    try:
        url = request.json["url"]

        download(url, input_path)

        results = []
        nudity = classify(input_path)
        results.append({"nudity": str(True),"score": "{0:.4f}".format(nudity)})
        results.append({"nudity": str(False), "score": "{0:.4f}".format(1-nudity)})

        return json.dumps(results), 200


    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


    finally:
        clean_all([
            input_path
        ])



if __name__ == '__main__':
    global upload_directory, model_directory
    global nsfw_net, caffe_transformer


    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    model_directory = '/src/nsfw_model/'
    create_directory(model_directory)

    model_url_prefix = "http://pretrained-models.auth-18b62333a540498882ff446ab602528b.storage.gra5.cloud.ovh.net/image/nsfw/"

    get_multi_model_bin([(
            model_url_prefix + 'deploy.prototxt', 
            model_directory + 'deploy.prototxt'
        ),(
            model_url_prefix + 'resnet_50_1by2_nsfw.caffemodel',
            model_directory + 'resnet_50_1by2_nsfw.caffemodel'
        )])
    
    nsfw_net, caffe_transformer = load_model()



    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)
