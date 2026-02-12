import json
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'mask_model.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'threshold.json')
FACE_MODEL_PROTO = os.path.join(MODEL_DIR, 'deploy.prototxt')
FACE_MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

IMG_SIZE = (224, 224)


def load_threshold() -> float:
    if not os.path.exists(THRESHOLD_PATH):
        return 0.5
    try:
        with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return float(data.get('without_mask_threshold', 0.5))
    except Exception:
        return 0.5


def is_with_mask_label(label: str) -> bool:
    return label.strip().lower() == 'with_mask'


def label_to_status(label: str) -> str:
    normalized = label.strip().lower().replace(' ', '_')
    if normalized == 'with_mask':
        return 'with_mask'
    if normalized in ('without_mask', 'no_mask'):
        return 'without_mask'
    if normalized.startswith('without') or normalized.startswith('no_'):
        return 'without_mask'
    return 'with_mask'


def status_to_model_label(status: str, class_names) -> str:
    for name in class_names:
        if label_to_status(name) == status:
            return name
    return class_names[0] if status == 'with_mask' else class_names[-1]


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    return np.expand_dims(face_resized, axis=0).astype('float32')


def confidence_from_threshold(prob_without_mask: float, threshold: float) -> float:
    t = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    p = min(max(float(prob_without_mask), 0.0), 1.0)
    if p >= t:
        conf = (p - t) / (1.0 - t)
    else:
        conf = (t - p) / t
    return float(min(max(conf, 0.0), 1.0))


def load_nose_cascade(model_dir: str):
    candidates = [
        os.path.join(model_dir, 'haarcascade_mcs_nose.xml'),
        os.path.join(model_dir, 'haarcascade_nose.xml'),
        os.path.join(cv2.data.haarcascades, 'haarcascade_mcs_nose.xml'),
        os.path.join(cv2.data.haarcascades, 'haarcascade_nose.xml'),
    ]
    for path in candidates:
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    return None


def detect_faces_dnn(net, frame_bgr: np.ndarray):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

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


def predict_mask(face_bgr: np.ndarray, model, class_names, without_mask_threshold: float, eye_cascade, smile_cascade, nose_cascade):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h >= 40 and w >= 40:
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
            return status_to_model_label('without_mask', class_names), 0.88
        if eyes_visible and (nose_visible or nose_cascade is None):
            return status_to_model_label('with_mask', class_names), 0.82
        if eyes_visible:
            return status_to_model_label('with_mask', class_names), 0.80

    x = preprocess_face(face_bgr)
    preds = model.predict(x, verbose=0)[0]

    if np.ndim(preds) == 0 or preds.shape == ():
        prob_class_1 = float(preds)
        if prob_class_1 >= without_mask_threshold:
            label = class_names[1]
        else:
            label = class_names[0]
        confidence = confidence_from_threshold(prob_class_1, without_mask_threshold)
    else:
        idx = int(np.argmax(preds))
        label = class_names[idx]
        confidence = float(preds[idx])

    return label, confidence


def smooth_prediction(history, current_label: str, current_confidence: float, state: dict, class_names):
    history.append((current_label, float(current_confidence)))

    with_weight = sum(c for l, c in history if label_to_status(l) == 'with_mask')
    without_weight = sum(c for l, c in history if label_to_status(l) == 'without_mask')
    total = max(with_weight + without_weight, 1e-6)

    if with_weight >= without_weight:
        candidate = status_to_model_label('with_mask', class_names)
        candidate_conf = with_weight / total
    else:
        candidate = status_to_model_label('without_mask', class_names)
        candidate_conf = without_weight / total

    if state['stable_label'] is None:
        state['stable_label'] = candidate
        state['stable_confidence'] = candidate_conf
        state['pending_label'] = None
        state['pending_count'] = 0
        return state['stable_label'], state['stable_confidence']

    if candidate == state['stable_label']:
        state['stable_confidence'] = candidate_conf
        state['pending_label'] = None
        state['pending_count'] = 0
        return state['stable_label'], state['stable_confidence']

    if state['pending_label'] == candidate:
        state['pending_count'] += 1
    else:
        state['pending_label'] = candidate
        state['pending_count'] = 1

    if state['pending_count'] >= 3 and candidate_conf >= 0.65:
        state['stable_label'] = candidate
        state['stable_confidence'] = candidate_conf
        state['pending_label'] = None
        state['pending_count'] = 0

    return state['stable_label'], state['stable_confidence']


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model not found. Train it first: python train.py')
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError('Missing labels.json. Train it first: python train.py')
    if not (os.path.exists(FACE_MODEL_PROTO) and os.path.exists(FACE_MODEL_WEIGHTS)):
        raise FileNotFoundError('Missing face detector files in model/. Run download_face_model.py')

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    without_mask_threshold = load_threshold()
    face_net = cv2.dnn.readNetFromCaffe(FACE_MODEL_PROTO, FACE_MODEL_WEIGHTS)
    eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml'))
    smile_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml'))
    nose_cascade = load_nose_cascade(MODEL_DIR)
    history = deque(maxlen=7)
    state = {
        'stable_label': None,
        'stable_confidence': 0.0,
        'pending_label': None,
        'pending_count': 0,
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Failed to open camera')

    print('Press Q to quit')

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes = detect_faces_dnn(face_net, frame)
        boxes = sorted(boxes, key=lambda r: r[2] * r[3], reverse=True)[:1]
        if not boxes:
            history.clear()
            state['stable_label'] = None
            state['stable_confidence'] = 0.0
            state['pending_label'] = None
            state['pending_count'] = 0

        for (x, y, w, h) in boxes:
            face = frame[y:y + h, x:x + w]
            raw_label, raw_confidence = predict_mask(
                face,
                model,
                class_names,
                without_mask_threshold,
                eye_cascade,
                smile_cascade,
                nose_cascade,
            )
            label, confidence = smooth_prediction(history, raw_label, raw_confidence, state, class_names)

            if is_with_mask_label(label):
                color = (0, 180, 0)
                text_label = 'Mask'
            else:
                color = (0, 0, 255)
                text_label = 'No Mask'

            text = f'{text_label}: {confidence * 100:.1f}%'
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Mask Detection', frame)

        if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
