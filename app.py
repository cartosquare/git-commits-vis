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

REPO_DIRNAME = '/Volumes/first/ml/terrain-context-deploy/beijing-demo'

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
    z = flask.request.args.get('z', '')
    x = flask.request.args.get('x', '')
    y = flask.request.args.get('y', '')
    return app.clf.classify_image(z, x, y)


class TerrainClassifier(object):
    default_args = {
        'caffe_model_def': (
            '{}/deploy.prototxt'.format(REPO_DIRNAME)),
        'caffe_model_weight': (
            '{}/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'image_subtraction_file': (
            '{}/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
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

    def __init__(self, caffe_model_def, caffe_model_weight, image_subtraction_file, tags_file, model_architecture_file, model_weights_file,
                 features_file, keys_file):
        logging.info('Loading net and associated files...')
        caffe.set_mode_cpu()

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

        # load caffe model
        self.net = caffe.Net(caffe_model_def, caffe_model_weight, caffe.TEST)

        # load the mean ImageNet image (as distributed with Caffe) for subtraction
        mu = np.load(image_subtraction_file)
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        print 'mean-subtracted values:', zip('BGR', mu)

        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

        ### set the size of the input
        # batch size = 1; 3-channel (BGR) images; image size is 227x227
        self.net.blobs['data'].reshape(1, 3, 227, 227)

        # load classify model
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

    def get_image_feature_from_url(self, image_url):
        image = caffe.io.load_image(image_url)
        transformed_image = self.transformer.preprocess('data', image)
        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = transformed_image

        ### forward calculation
        self.net.forward()
        feature = np.transpose(self.net.blobs['fc7'].data[0])
        feature = np.array([feature])
        feature = feature.astype('float32')
        return feature

    def classify_image(self, z, x, y):
        image_url = 'http://mt2.google.cn/vt/lyrs=s&hl=zh-CN&gl=cn&x=%s&y=%s&z=%s&scale=1' % (x, y, z)
        print image_url

        features = self.get_image_feature_from_url(image_url)
        # features = self.get_image_feature(x + '_' + y)

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
