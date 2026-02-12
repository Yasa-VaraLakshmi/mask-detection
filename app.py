import base64
import io
import json
import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'mask_model.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'threshold.json')
FACE_MODEL_PROTO = os.path.join(MODEL_DIR, 'deploy.prototxt')
FACE_MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

IMG_SIZE = (224, 224)
MIN_CONFIDENCE_FOR_DECISION = 0.80
PREDICTION_STALE_SECONDS = 8.0

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError('Model not found. Train it first: python train.py')

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

without_mask_threshold = 0.5
if os.path.exists(THRESHOLD_PATH):
    try:
        with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
            threshold_data = json.load(f)
        without_mask_threshold = float(threshold_data.get('without_mask_threshold', 0.5))
    except Exception:
        without_mask_threshold = 0.5

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
)
smile_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml')
)


def load_nose_cascade():
    candidates = [
        os.path.join(MODEL_DIR, 'haarcascade_mcs_nose.xml'),
        os.path.join(MODEL_DIR, 'haarcascade_nose.xml'),
        os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_nose.xml'),
        os.path.join(cv2.data.haarcascades, 'haarcascade_nose.xml'),
    ]
    for path in candidates:
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    return None


nose_cascade = load_nose_cascade()

face_net = None
if os.path.exists(FACE_MODEL_PROTO) and os.path.exists(FACE_MODEL_WEIGHTS):
    face_net = cv2.dnn.readNetFromCaffe(FACE_MODEL_PROTO, FACE_MODEL_WEIGHTS)


class SpeechManager:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            text = self.queue.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as exc:
                print(f'Voice error: {exc}')

    def speak(self, text: str):
        self.queue.put(text)


speech = SpeechManager()
last_spoken_label = None
last_spoken_time = 0.0
last_prediction = None
last_prediction_time = 0.0
prediction_history = deque(maxlen=7)
stable_label = None
stable_confidence = 0.0
pending_label = None
pending_count = 0


def label_to_status(label: str) -> str:
    normalized = label.strip().lower().replace(' ', '_')
    if normalized == 'with_mask':
        return 'with_mask'
    if normalized in ('without_mask', 'no_mask'):
        return 'without_mask'
    if normalized.startswith('without') or normalized.startswith('no_'):
        return 'without_mask'
    return 'with_mask'


def is_with_mask(label: str) -> bool:
    return label_to_status(label) == 'with_mask'


def status_to_model_label(status: str) -> str:
    for name in class_names:
        if label_to_status(name) == status:
            return name
    return class_names[0] if status == 'with_mask' else class_names[-1]


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    x = np.expand_dims(face_resized, axis=0).astype('float32') / 255.0
    return x


def confidence_from_threshold(prob_without_mask: float, threshold: float) -> float:
    t = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    p = min(max(float(prob_without_mask), 0.0), 1.0)
    if p >= t:
        conf = (p - t) / (1.0 - t)
    else:
        conf = (t - p) / t
    return float(min(max(conf, 0.0), 1.0))


def predict_face_by_visibility(face_bgr: np.ndarray):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return None

    upper = gray[: int(h * 0.6), :]
    lower = gray[int(h * 0.45):, :]
    middle = gray[int(h * 0.3): int(h * 0.8), :]

    eyes = eye_cascade.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=4, minSize=(14, 14))
    mouths = smile_cascade.detectMultiScale(lower, scaleFactor=1.7, minNeighbors=20, minSize=(24, 16))
    noses = ()
    if nose_cascade is not None:
        noses = nose_cascade.detectMultiScale(middle, scaleFactor=1.2, minNeighbors=5, minSize=(18, 18))

    eyes_visible = len(eyes) >= 1
    mouth_visible = len(mouths) >= 1
    nose_visible = len(noses) >= 1

    if eyes_visible and mouth_visible and (nose_visible or nose_cascade is None):
        return status_to_model_label('without_mask'), 0.88
    if eyes_visible and (nose_visible or nose_cascade is None):
        return status_to_model_label('with_mask'), 0.82
    if eyes_visible:
        return status_to_model_label('with_mask'), 0.80
    return None


def predict_face(face_bgr: np.ndarray):
    visibility_prediction = predict_face_by_visibility(face_bgr)
    if visibility_prediction is not None:
        return visibility_prediction

    x = preprocess_face(face_bgr)
    preds = model.predict(x, verbose=0)[0]

    if np.ndim(preds) == 0 or preds.shape == ():
        prob_without_mask = float(preds)
        if prob_without_mask >= without_mask_threshold:
            label = class_names[1]
        else:
            label = class_names[0]
        confidence = confidence_from_threshold(prob_without_mask, without_mask_threshold)
    else:
        idx = int(np.argmax(preds))
        label = class_names[idx]
        confidence = float(preds[idx])

    return label, confidence


def update_stable_prediction(raw_label: str, raw_confidence: float):
    global stable_label, stable_confidence, pending_label, pending_count

    prediction_history.append((raw_label, float(raw_confidence)))

    with_weight = sum(c for l, c in prediction_history if is_with_mask(l))
    without_weight = sum(c for l, c in prediction_history if not is_with_mask(l))
    total = max(with_weight + without_weight, 1e-6)

    if with_weight >= without_weight:
        candidate = status_to_model_label('with_mask')
        candidate_conf = with_weight / total
    else:
        candidate = status_to_model_label('without_mask')
        candidate_conf = without_weight / total

    if stable_label is None:
        stable_label = candidate
        stable_confidence = candidate_conf
        pending_label = None
        pending_count = 0
        return stable_label, stable_confidence

    if candidate == stable_label:
        stable_confidence = candidate_conf
        pending_label = None
        pending_count = 0
        return stable_label, stable_confidence

    if pending_label == candidate:
        pending_count += 1
    else:
        pending_label = candidate
        pending_count = 1

    if pending_count >= 3 and candidate_conf >= 0.65:
        stable_label = candidate
        stable_confidence = candidate_conf
        pending_label = None
        pending_count = 0

    return stable_label, stable_confidence


def reset_prediction_state():
    global stable_label, stable_confidence, pending_label, pending_count
    prediction_history.clear()
    stable_label = None
    stable_confidence = 0.0
    pending_label = None
    pending_count = 0


def detect_faces_haar(gray: np.ndarray):
    h, w = gray.shape[:2]
    min_size = int(min(h, w) * 0.12)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(min_size, min_size),
    )


def detect_faces_dnn(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < 0.5:
            continue
        x1, y1, x2, y2 = (
            detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        ).astype('int')
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def maybe_speak(results):
    global last_spoken_label, last_spoken_time
    if not results:
        return

    first = results[0]
    label = first['label']
    confidence = float(first['confidence'])
    if confidence < MIN_CONFIDENCE_FOR_DECISION:
        return

    now = time.time()
    if label != last_spoken_label or (now - last_spoken_time) > 3:
        if is_with_mask(label):
            text = 'The person is wearing a mask'
        else:
            text = 'The person is not wearing a mask'
        speech.speak(text)
        last_spoken_label = label
        last_spoken_time = now


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    global last_prediction, last_prediction_time

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    img_data = data['image']
    if ',' in img_data:
        img_data = img_data.split(',', 1)[1]

    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if face_net is not None:
        faces = detect_faces_dnn(bgr)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_haar(gray)

    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[:1]

    results = []
    for (x, y, w, h) in faces:
        face = bgr[y:y + h, x:x + w]
        raw_label, raw_conf = predict_face(face)
        label, conf = update_stable_prediction(raw_label, raw_conf)
        results.append({'label': label, 'confidence': conf})
        last_prediction = {'label': label, 'confidence': conf}
        last_prediction_time = time.time()

    if not results:
        last_prediction = None
        last_prediction_time = 0.0
        reset_prediction_state()

    maybe_speak(results)
    return jsonify({'ok': True, 'results': results})


@app.route('/command', methods=['POST'])
def command():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').lower()

    if not text:
        return jsonify({'error': 'No command text'}), 400

    asks_mask_status = (
        'mask' in text and (
            'wear' in text or 'wearing' in text or 'status' in text or 'detect' in text
        )
    )

    if asks_mask_status:
        if last_prediction is None or (time.time() - last_prediction_time) > PREDICTION_STALE_SECONDS:
            reply = 'No detection yet. Please look at the camera.'
        else:
            label = last_prediction['label']
            confidence = float(last_prediction['confidence'])
            conf_pct = round(confidence * 100.0, 1)

            if confidence < MIN_CONFIDENCE_FOR_DECISION:
                reply = f'I am not confident enough to decide yet. Current confidence is {conf_pct} percent. Please face the camera clearly.'
            elif is_with_mask(label):
                reply = f'The person in front of the cam is wearing a mask. Confidence {conf_pct} percent.'
            else:
                reply = f'The person in front of the cam is not wearing a mask. Confidence {conf_pct} percent.'

        speech.speak(reply)
        return jsonify({'reply': reply})

    reply = 'Command not recognized. You can ask if the person is wearing a mask.'
    speech.speak(reply)
    return jsonify({'reply': reply})


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
