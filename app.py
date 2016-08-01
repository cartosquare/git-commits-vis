import os
import time
import cPickle
import datetime
import logging
import flask
from flask_cors import CORS, cross_origin
from flask import send_file
from flask import request
import werkzeug
import optparse
import re
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import json
import cPickle
import urllib2
from sklearn.externals import joblib
import skimage.io
from keras.models import model_from_json

import sys
caffe_root = '../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import leveldb
from caffe.proto import caffe_pb2

REPO_DIRNAME = '/Volumes/first/ml/terrain-context-deploy/beijing-demo'
#REPO_DIRNAME = '/Volumes/first/bj_demo'

# Obtain the flask app object
app = flask.Flask(__name__)
CORS(app)


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
def classify():
    z = flask.request.args.get('z', '')
    x = flask.request.args.get('x', '')
    y = flask.request.args.get('y', '')
    return app.clf.classify_image(z, x, y)


@app.route('/similar', methods=['GET'])
def similar():
    z = flask.request.args.get('z', '')
    x = flask.request.args.get('x', '')
    y = flask.request.args.get('y', '')
    n = flask.request.args.get('limit', '')
    r = flask.request.args.get('region', '')
    features = app.clf.get_tile_feature(z, x, y)
    url = 'http://localhost:5566/similar/v2'
    feature_json = {}
    feature_json['feature'] = features[0].tolist()

    data = urllib.urlencode({
        'limit': n,
        'region': r,
        'feature': json.dumps(feature_json)
    })

    content = urllib2.urlopen(url=url, data=data).read()
    return flask.Response(content, mimetype='application/json')


@app.route('/terrain-context', methods=['GET'])
def terrain_context():
    z = flask.request.args.get('z', '')
    x = flask.request.args.get('x', '')
    y = flask.request.args.get('y', '')
    n = flask.request.args.get('limit', '')
    r = flask.request.args.get('region', '')
    return app.clf.classify_and_search_similar(z, x, y, n, r)


@app.route('/v2/terrain-context', methods=['POST'])
def terrain_context_v2():
    # print 'request incoming...'
    z = int(request.form['level'])

    x = float(request.form['x'])
    y = float(request.form['y'])
    n = int(request.form['limit'])
    r = str(request.form['region'])
    base64_image = request.form['image']
    # print z, x, y, n, r
    return app.clf.classify_and_search_similar(base64_image, z, x, y, n, r)


@app.route('/crop', methods=['GET'])
def merge_crop():
    z = flask.request.args.get('z', '')
    x = flask.request.args.get('x', '')
    y = flask.request.args.get('y', '')
    return app.clf.merge_and_crop_image(int(z), float(x), float(y))


class TerrainClassifier(object):
    default_args = {
        'caffe_model_def': (
            '{}/deploy.prototxt'.format(REPO_DIRNAME)),
        'caffe_model_weight': (
            '{}/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'image_subtraction_file': (
            '{}/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'tags_file': (
            '{}/tags_67.csv'.format(REPO_DIRNAME)),
        'model_architecture_file': (
            '{}/tags_67_model_architecture_bvlc_fc7.json'.format(REPO_DIRNAME)),
        'model_weights_file': (
            '{}/tags_67_model_weights_bvlc_fc7.h5'.format(REPO_DIRNAME))
    }

    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, caffe_model_def, caffe_model_weight, image_subtraction_file, tags_file, model_architecture_file, model_weights_file):
        self.res = 0.597164283477783
        self.tilesize = 256
        self.tile_extent = self.tilesize * self.res
        self.world_originalx = -20037508.342787
        self.world_originaly = 20037508.342787

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

    def merge_and_crop_image(self, z, x, y):
        row = int((x - self.world_originalx) / self.tile_extent)
        col = int((self.world_originaly - y) / self.tile_extent)
        x_center = self.world_originalx + self.tile_extent * (row + 0.5)
        y_center = self.world_originaly - self.tile_extent * (col + 0.5)

        if x <= x_center:
            if y >= y_center:
                # left-top corner
                # print 'left-top corner'
                r1 = r3 = row - 1
                r2 = r4 = row
                c1 = c2 = col - 1
                c3 = c4 = col
            else:
                # left-bottom corner
                # print 'left-bottom corner'
                r1 = r3 = row - 1
                r2 = r4 = row
                c1 = c2 = col
                c3 = c4 = col + 1
        else:
            if y >= y_center:
                # right-top corner
                # print 'right-top corner'
                r1 = r3 = row
                r2 = r4 = row + 1
                c1 = c2 = col - 1
                c3 = c4 = col
            else:
                # right-bottom corner
                # print 'right-bottom corner'
                r1 = r3 = row
                r2 = r4 = row + 1
                c1 = c2 = col
                c3 = c4 = col + 1

        url1 = 'http://mt2.google.cn/vt/lyrs=s&hl=zh-CN&gl=cn&x=%s&y=%s&z=%s&scale=1' % (r1, c1, z)
        url2 = 'http://mt2.google.cn/vt/lyrs=s&hl=zh-CN&gl=cn&x=%s&y=%s&z=%s&scale=1' % (r2, c2, z)
        url3 = 'http://mt2.google.cn/vt/lyrs=s&hl=zh-CN&gl=cn&x=%s&y=%s&z=%s&scale=1' % (r3, c3, z)
        url4 = 'http://mt2.google.cn/vt/lyrs=s&hl=zh-CN&gl=cn&x=%s&y=%s&z=%s&scale=1' % (r4, c4, z)

        image1 = Image.open(StringIO.StringIO(urllib.urlopen(url1).read()))
        # image1.save("img1.png", "PNG")

        image2 = Image.open(StringIO.StringIO(urllib.urlopen(url2).read()))
        # image2.save("img2.png", "PNG")

        image3 = Image.open(StringIO.StringIO(urllib.urlopen(url3).read()))
        # image3.save("img3.png", "PNG")

        image4 = Image.open(StringIO.StringIO(urllib.urlopen(url4).read()))
        # image4.save("img4.png", "PNG")

        merged_image = Image.new('RGB', (512, 512))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (256, 0))
        merged_image.paste(image3, (0, 256))
        merged_image.paste(image4, (256, 256))
        # merged_image.save("img_merge.png", "PNG")

        # crop
        crop_originx = x - self.tile_extent / 2.0
        crop_originy = y + self.tile_extent / 2.0
        # print 'crop map origin: ', crop_originx, crop_originy

        merged_image_originalx = self.world_originalx + self.tile_extent * r1
        merged_image_originaly = self.world_originaly - self.tile_extent * c1
        # print 'merged image origin: ', merged_image_originalx, merged_image_originaly

        crop_x = int((crop_originx - merged_image_originalx) / self.res)
        crop_y = int((merged_image_originaly - crop_originy) / self.res)
        # print 'crop pixel origin: ', crop_x, crop_y
        if crop_x + 256 > 512:
            print 'invalid crop x'
        if crop_y + 256 > 512:
            print 'invalid crop y'

        crop_image = merged_image.crop((crop_x, crop_y, 256 + crop_x, 256 + crop_y))
        # crop_image.save("img_crop.png", "PNG")

        img = skimage.img_as_float(crop_image).astype(np.float32)
        # print img.ndim

        img = np.tile(img, (1, 1, 3))

        return img

    def get_tile_feature(self, base64_image, z, x, y, layer_name):
        imgstr = re.search(r'base64,(.*)', base64_image).group(1)
        image = Image.open(StringIO.StringIO(imgstr.decode('base64')))
        # image.save("back.png", "PNG")

        img = skimage.img_as_float(image).astype(np.float32)
        img = np.tile(img, (1, 1, 3))

        transformed_image = self.transformer.preprocess('data', img)
        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = transformed_image

        ### forward calculation
        self.net.forward()

        # collect the layers
        features = {}
        for layer in layer_name:
            feat = np.transpose(self.net.blobs[layer].data[0])
            feat = np.array([feat])
            feat = feat.astype('float32')
            features[layer] = feat

        return features

    def classify_image(self, z, x, y):
        features = self.get_tile_feature(z, x, y, ['fc7'])
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
            res['prob'] = str(a[ii])
            result['labels'].append(res)

        # return json.dumps(result, ensure_ascii=False)
        return flask.jsonify(**result)

    def classify_and_search_similar(self, base64_image, z, x, y, limit, region):
        features = self.get_tile_feature(base64_image, z, x, y, ['fc7', 'fc7'])

        # classify
        pred = self.model.predict(features['fc7'])
        a = list(pred[0])
        b = sorted(range(len(a)), key=lambda i: a[i])[-5:]
        result = {}
        result['labels'] = []
        for i in range(1, 6):
            ii = b[5 - i]
            res = {}
            res['label'] = self.labels_lookup[ii]
            res['prob'] = str(a[ii])
            result['labels'].append(res)
        # search similar
        url = 'http://localhost:5566/similar/v2'
        feature_json = {}
        feature_json['feature'] = features['fc7'][0].tolist()
        data = urllib.urlencode({
            'limit': limit,
            'region': region,
            'feature': json.dumps(feature_json)
        })

        content = urllib2.urlopen(url=url, data=data).read()
        result['similars'] = content
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
