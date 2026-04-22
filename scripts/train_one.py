"""
Train a single model with a single seed.
Usage: python train_one.py --model {vgg16,vit,hybrid_gated,a2wnet} --seed 42
Writes: results/runs/{model}_seed{seed}.json
"""
import os
import sys
import json
import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf

# ── Thread limiting ────────────────────────────────────────────────────────────
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'

# ── Parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    choices=['vgg16', 'vit', 'hybrid_gated', 'a2wnet'])
parser.add_argument('--seed', required=True, type=int)
args = parser.parse_args()
MODEL_NAME = args.model
SEED = args.seed

# ── Set seeds ──────────────────────────────────────────────────────────────────
# PYTHONHASHSEED is set by train_seeds.py before spawning this process.
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)  # sets random + numpy + tf in one call

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)  # for utils.py

# ── Dataset loading ────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_dataset(dataset_dir):
    records = []
    for label in sorted(os.listdir(dataset_dir)):  # sorted for cross-platform determinism
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({
                    'filename': os.path.join(dataset_dir, label, fname),
                    'label': label
                })
    return pd.DataFrame(records)


dataset_dir = os.path.join(PROJECT_ROOT, 'dataset')
df = load_dataset(dataset_dir)
NUM_CLASSES = len(df['label'].unique())

# Test split ALWAYS fixed at random_state=42 — must not vary with seed
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)


def make_generators(preprocess_fn, seed):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filename', y_col='label',
        target_size=(224, 224), class_mode='categorical',
        batch_size=32, shuffle=True, seed=seed, subset='training'
    )
    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filename', y_col='label',
        target_size=(224, 224), class_mode='categorical',
        batch_size=32, shuffle=False, seed=seed, subset='validation'
    )
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='filename', y_col='label',
        target_size=(224, 224), color_mode='rgb',
        class_mode='categorical', batch_size=32, shuffle=False, verbose=0
    )
    return train_gen, valid_gen, test_gen


# ── Output paths ───────────────────────────────────────────────────────────────
os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'runs'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'weights'), exist_ok=True)

out_path = os.path.join(PROJECT_ROOT, 'results', 'runs',
                        f'{MODEL_NAME}_seed{SEED}.json')
weights_path = os.path.join(PROJECT_ROOT, 'results', 'weights',
                             f'{MODEL_NAME}_seed{SEED}.weights.h5')


# ── Model-specific training ────────────────────────────────────────────────────

if MODEL_NAME == 'vgg16':
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.optimizers import Adam
    from baselines.vgg16 import VGG16Classifier

    train_gen, valid_gen, test_gen = make_generators(preprocess_input, SEED)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    model = VGG16Classifier(num_classes=NUM_CLASSES, freeze_base=True)
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== VGG16 seed={SEED} Phase 1 ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=20, callbacks=callbacks, verbose=1)

    model.unfreeze_top_blocks(num_blocks=2)
    model.compile(optimizer=Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== VGG16 seed={SEED} Phase 2 ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=30, callbacks=callbacks, verbose=1)

    eval_results = model.evaluate(test_gen, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    print('metrics_names:', model.metrics_names)
    test_loss = float(eval_results[0])
    test_acc  = float(eval_results[1])


elif MODEL_NAME == 'vit':
    import keras_hub
    from tensorflow.keras.optimizers import AdamW
    from baselines.vit import ViTClassifier

    def rescale(x): return x / 255.0
    train_gen, valid_gen, test_gen = make_generators(rescale, SEED)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    model = ViTClassifier(num_classes=NUM_CLASSES, freeze_base=True)
    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== ViT seed={SEED} Phase 1 ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=20, callbacks=callbacks, verbose=1)

    model.unfreeze_last_blocks(num_blocks=4)
    model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== ViT seed={SEED} Phase 2 ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=30, callbacks=callbacks, verbose=1)

    eval_results = model.evaluate(test_gen, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    print('metrics_names:', model.metrics_names)
    test_loss = float(eval_results[0])
    test_acc  = float(eval_results[1])


elif MODEL_NAME == 'hybrid_gated':
    from tensorflow.keras.optimizers import AdamW
    from contributions.hybrid_gated import HybridGatedModel

    def raw_identity(x): return x
    train_gen, valid_gen, test_gen = make_generators(raw_identity, SEED)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    model = HybridGatedModel(num_classes=NUM_CLASSES, freeze_base=True)
    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== HybridGated seed={SEED} Phase 1 ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=15, callbacks=callbacks, verbose=1)

    model.unfreeze_last_blocks(vit_blocks=4, vgg_blocks=2)
    model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== HybridGated seed={SEED} Phase 2 ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=20, callbacks=callbacks, verbose=1)

    eval_results = model.evaluate(test_gen, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    print('metrics_names:', model.metrics_names)
    test_loss = float(eval_results[0])
    test_acc  = float(eval_results[1])


elif MODEL_NAME == 'a2wnet':
    from tensorflow.keras.optimizers import AdamW
    from contributions.hybrid_contrastive import (SupervisedContrastiveLoss,
                                    A2WNet_Contrastive,
                                    multi_loss_generator)

    def raw_identity(x): return x
    base_train_gen, base_valid_gen, base_test_gen = make_generators(raw_identity, SEED)

    train_gen = multi_loss_generator(base_train_gen)
    valid_gen = multi_loss_generator(base_valid_gen)

    loss_funcs = {
        'predictions': 'categorical_crossentropy',
        'features': SupervisedContrastiveLoss(temperature=0.1)
    }
    loss_weights = {'predictions': 1.0, 'features': 0.15}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    model = A2WNet_Contrastive(num_classes=NUM_CLASSES, freeze_base=True)
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_funcs, loss_weights=loss_weights,
        metrics={'predictions': ['accuracy']}
    )

    print(f"=== A2WNet seed={SEED} Phase 1 ===")
    model.fit(
        train_gen, steps_per_epoch=len(base_train_gen),
        validation_data=valid_gen, validation_steps=len(base_valid_gen),
        epochs=15, callbacks=callbacks, verbose=1
    )

    model.unfreeze_last_blocks()
    model.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=loss_funcs, loss_weights=loss_weights,
        metrics={'predictions': ['accuracy']}
    )

    print(f"=== A2WNet seed={SEED} Phase 2 ===")
    model.fit(
        train_gen, steps_per_epoch=len(base_train_gen),
        validation_data=valid_gen, validation_steps=len(base_valid_gen),
        epochs=20, callbacks=callbacks, verbose=1
    )

    # Evaluate on test set — extract predictions_accuracy from named metrics
    test_gen_multi = multi_loss_generator(base_test_gen)
    eval_results = model.evaluate(
        test_gen_multi, steps=len(base_test_gen), verbose=0
    )
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    print('metrics_names:', model.metrics_names)
    # Scan by substring — handles Keras 2/3 naming differences
    test_loss = next((float(v) for k, v in metrics_dict.items()
                      if 'predictions' in k and 'loss' in k), float(eval_results[1]))
    test_acc  = next((float(v) for k, v in metrics_dict.items()
                      if 'accuracy' in k), float(eval_results[-1]))


# ── Save result ────────────────────────────────────────────────────────────────
result_dict = {
    'model':         MODEL_NAME,
    'seed':          SEED,
    'test_loss':     float(test_loss),
    'test_accuracy': float(test_acc),
    'all_metrics':   {k: float(v) for k, v in metrics_dict.items()}
}
with open(out_path, 'w') as f:
    json.dump(result_dict, f, indent=2)

print(f"\n=== RESULT: {MODEL_NAME} seed={SEED} "
      f"| loss={test_loss:.5f} | acc={test_acc*100:.2f}% ===")
print(f"Saved to {out_path}")
