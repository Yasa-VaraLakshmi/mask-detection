import os
import cv2
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = ROOT
WITH_DIR = os.path.join(ROOT, 'dataset', 'with_mask')
WITHOUT_DIR = os.path.join(ROOT, 'dataset', 'without_mask')

os.makedirs(WITH_DIR, exist_ok=True)
os.makedirs(WITHOUT_DIR, exist_ok=True)

# Collect images in root (exclude dataset folder)
images = [
    f for f in os.listdir(SRC_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(SRC_DIR, f))
]
images.sort()

if not images:
    print('No images found in root to label.')
    raise SystemExit(0)

print('Labeling controls:')
print('  w = with_mask')
print('  n = without_mask')
print('  s = skip')
print('  q = quit')

idx = 0
while idx < len(images):
    img_name = images[idx]
    img_path = os.path.join(SRC_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f'Skipping unreadable: {img_name}')
        idx += 1
        continue

    display = img.copy()
    cv2.putText(display, f'{idx+1}/{len(images)}: {img_name}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, 'w=with | n=without | s=skip | q=quit', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Label Images', display)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    if key == ord('s'):
        idx += 1
        continue
    if key == ord('w'):
        dest = os.path.join(WITH_DIR, img_name)
        shutil.move(img_path, dest)
        print(f'WITH: {img_name}')
        idx += 1
        continue
    if key == ord('n'):
        dest = os.path.join(WITHOUT_DIR, img_name)
        shutil.move(img_path, dest)
        print(f'WITHOUT: {img_name}')
        idx += 1
        continue

    print('Invalid key, use w/n/s/q')

cv2.destroyAllWindows()
print('Done.')
