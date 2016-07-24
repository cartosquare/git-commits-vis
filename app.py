import os
import time
import cPickle
import datetime
import logging
import flask
from flask import send_file
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import json
from keras.models import model_from_json

import sys
caffe_root = '../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import leveldb
from caffe.proto import caffe_pb2

REPO_DIRNAME = '/Users/xuxiang/ml/terrain-context/dist/beijing-demo'

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/tile/<int:x>/<int:y>')
def tile(x, y):
    filename = '{}/google/{}/{}.jpg'.format(REPO_DIRNAME, y, x)
    if os.path.exists(filename):
        return send_file(filename, mimetype='image/jpg')
    else:
        f = {}
        f['err'] = 1
        return flask.jsonify(**f)


@app.route('/classify', methods=['GET'])
def classify_url():
    imagename = flask.request.args.get('image', '')
    print imagename
    return app.clf.classify_image(imagename)


class TerrainClassifier(object):
    default_args = {
        'tags_file': (
            '{}/tags.csv'.format(REPO_DIRNAME)),
        'model_architecture_file': (
            '{}/model_architecture_bvlc_fc7.json'.format(REPO_DIRNAME)),
        'model_weights_file': (
            '{}/model_weights_bvlc_fc7.h5'.format(REPO_DIRNAME)),
        'features_file': (
            '{}/L18_deep_features_bvlc'.format(REPO_DIRNAME)),
        'keys_file': (
            '{}/L18_keys'.format(REPO_DIRNAME))
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, tags_file, model_architecture_file, model_weights_file,
                 features_file, keys_file):
        logging.info('Loading net and associated files...')

        # label-lookup
        self.labels_lookup = []
        count = 0
        with open(tags_file) as f:
            for line in f:
                (tag, tag_cn, label) = line.strip().split(',')
                if count != int(label):
                    print 'invalid label loopup!'
                self.labels_lookup.append(tag_cn)
                print count, tag_cn
                count = count + 1

        # load model
        self.model = model_from_json(open(model_architecture_file).read())
        self.model.load_weights(model_weights_file)

        # finally, before it can be used, the model shall be compiled.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # open feature database
        self.features_db = leveldb.LevelDB(features_file)

        # open key database
        self.keys_db = leveldb.LevelDB(keys_file)

    def get_image_feature(self, image_name):
        key = self.keys_db.Get(image_name)
        datum = caffe_pb2.Datum.FromString(self.features_db.Get(key))
        data = caffe.io.datum_to_array(datum)
        feat = np.transpose(data[:, 0])[0]
        feat = np.array([feat])
        feat = feat.astype('float32')
        return feat

    def classify_image(self, image):
        features = self.get_image_feature(image)
        pred = self.model.predict(features)
        a = list(pred[0])
        b = sorted(range(len(a)), key=lambda i: a[i])[-5:]
        result = {}
        result['labels'] = []
        for i in range(1, 6):
            ii = b[5 - i]
            res = {}
            res['label'] = self.labels_lookup[ii]
            res['prob'] = a[ii]
            result['labels'].append(res)

        # return json.dumps(result, ensure_ascii=False)
        return flask.jsonify(**result)


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)

    opts, args = parser.parse_args()

    # Initialize classifier + warm start by forward for allocation
    app.clf = TerrainClassifier(**TerrainClassifier.default_args)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
