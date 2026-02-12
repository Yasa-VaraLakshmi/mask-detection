import json
import os

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'dataset')
MODEL_DIR = os.path.join(ROOT, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'threshold.json')

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print('Classes:', class_names)

labels_path = os.path.join(MODEL_DIR, 'labels.json')
with open(labels_path, 'w', encoding='utf-8') as f:
    json.dump(class_names, f)

class_counts = []
for name in class_names:
    folder = os.path.join(DATA_DIR, name)
    count = len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    class_counts.append(count)

if sum(class_counts) > 0:
    total = float(sum(class_counts))
    class_weight = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts) if c > 0}
else:
    class_weight = None

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y)).cache().shuffle(512).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'mask_model.keras'),
        save_best_only=True,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
)

# Threshold calibration on validation set using balanced accuracy.
probs = model.predict(val_ds, verbose=0).reshape(-1)
y_true = []
for _, y_batch in val_ds:
    y_true.extend(y_batch.numpy().reshape(-1).tolist())
y_true = np.array(y_true, dtype=np.float32)

best_thr = 0.5
best_score = -1.0
for thr in np.linspace(0.2, 0.8, 61):
    y_pred = (probs >= thr).astype(np.float32)
    pos_mask = y_true == 1.0
    neg_mask = y_true == 0.0
    tpr = float((y_pred[pos_mask] == 1.0).mean()) if pos_mask.any() else 0.0
    tnr = float((y_pred[neg_mask] == 0.0).mean()) if neg_mask.any() else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    if bal_acc > best_score:
        best_score = bal_acc
        best_thr = float(thr)

with open(THRESHOLD_PATH, 'w', encoding='utf-8') as f:
    json.dump({'without_mask_threshold': best_thr, 'balanced_accuracy': best_score}, f)
print(f'Calibrated threshold: {best_thr:.3f} (balanced_acc={best_score:.3f})')

model.save(os.path.join(MODEL_DIR, 'mask_model.keras'))
print('Saved model to model/mask_model.keras')
