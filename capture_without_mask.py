import os
import time
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'dataset', 'without_mask')
os.makedirs(OUT_DIR, exist_ok=True)

# Prefer DirectShow on Windows to avoid MSMF issues
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError('Failed to open webcam')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Find next image index
existing = [f for f in os.listdir(OUT_DIR) if f.lower().endswith('.png')]
max_idx = -1
for name in existing:
    if name.startswith('image_') and name.endswith('.png'):
        try:
            idx = int(name.replace('image_', '').replace('.png', ''))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass

counter = max_idx + 1
print('Press SPACE to capture, ESC to exit')

last_capture = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Capture Without Mask', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    if key == 32:
        now = time.time()
        if now - last_capture < 0.2:
            continue
        img_name = f'image_{counter}.png'
        out_path = os.path.join(OUT_DIR, img_name)
        cv2.imwrite(out_path, frame)
        print(f'Saved {img_name}')
        counter += 1
        last_capture = now

cap.release()
cv2.destroyAllWindows()
