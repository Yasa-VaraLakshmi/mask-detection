import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

PROTO_URL = 'https://raw.githubusercontent.com/opencv/opencv/3.4.0/samples/dnn/face_detector/deploy.prototxt'
WEIGHTS_URL = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'

proto_path = os.path.join(MODEL_DIR, 'deploy.prototxt')
weights_path = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

if not os.path.exists(proto_path):
    print('Downloading deploy.prototxt...')
    urllib.request.urlretrieve(PROTO_URL, proto_path)

if not os.path.exists(weights_path):
    print('Downloading res10_300x300_ssd_iter_140000.caffemodel...')
    urllib.request.urlretrieve(WEIGHTS_URL, weights_path)

print('Done.')
