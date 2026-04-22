"""
ImageNet-ablation: train VGG16 or A2WNet from scratch (no pretrained weights).
Same architecture, same epochs/LR schedule as the pretrained runs.

Usage: python train_ablation_one.py --model {vgg16_scratch,a2wnet_scratch} --seed 42
Writes: results/ablation_runs/{model}_seed{seed}.json
"""
import os
import sys
import json
import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'

# ── Args & seeds ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    choices=['vgg16_scratch', 'a2wnet_scratch'])
parser.add_argument('--seed', required=True, type=int)
args = parser.parse_args()
MODEL_NAME = args.model
SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# ── Dataset ────────────────────────────────────────────────────────────────────
def load_dataset(dataset_dir):
    records = []
    for label in sorted(os.listdir(dataset_dir)):
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

# Test split ALWAYS fixed — must match the pretrained runs for valid comparison
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
os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'ablation_runs'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'ablation_weights'), exist_ok=True)

out_path = os.path.join(PROJECT_ROOT, 'results', 'ablation_runs',
                        f'{MODEL_NAME}_seed{SEED}.json')
weights_path = os.path.join(PROJECT_ROOT, 'results', 'ablation_weights',
                             f'{MODEL_NAME}_seed{SEED}.weights.h5')


# ── VGG16 from scratch ─────────────────────────────────────────────────────────
if MODEL_NAME == 'vgg16_scratch':
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                         Dropout, BatchNormalization)

    class VGG16Scratch(Model):
        """Identical to VGG16Classifier but weights=None (random init)."""
        def __init__(self, num_classes, dropout_rate=0.5):
            super().__init__()
            # weights=None → random initialization, no ImageNet
            self.base   = VGG16(weights=None, include_top=False,
                                input_shape=(224, 224, 3))
            self.pool   = GlobalAveragePooling2D()
            self.bn1    = BatchNormalization()
            self.dense1 = Dense(512, activation='relu')
            self.drop1  = Dropout(dropout_rate)
            self.bn2    = BatchNormalization()
            self.dense2 = Dense(256, activation='relu')
            self.drop2  = Dropout(dropout_rate)
            self.out    = Dense(num_classes, activation='softmax')

        def call(self, x, training=False):
            x = self.base(x, training=training)  # train BN layers fully
            x = self.pool(x)
            x = self.bn1(x, training=training)
            x = self.dense1(x)
            x = self.drop1(x, training=training)
            x = self.bn2(x, training=training)
            x = self.dense2(x)
            x = self.drop2(x, training=training)
            return self.out(x)

    train_gen, valid_gen, test_gen = make_generators(vgg_preprocess, SEED)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    # No Phase 1/2 split — train the whole network end-to-end from scratch
    # (freezing random weights in Phase 1 would be meaningless)
    model = VGG16Scratch(num_classes=NUM_CLASSES)
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== VGG16-scratch seed={SEED} Phase 1 (full network, lr=1e-3) ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=20, callbacks=callbacks, verbose=1)

    model.compile(optimizer=Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"=== VGG16-scratch seed={SEED} Phase 2 (full network, lr=1e-5) ===")
    model.fit(train_gen, validation_data=valid_gen,
              epochs=30, callbacks=callbacks, verbose=1)

    eval_results = model.evaluate(test_gen, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    print('metrics_names:', model.metrics_names)
    test_loss = float(eval_results[0])
    test_acc  = float(eval_results[1])


# ── A2WNet from scratch ────────────────────────────────────────────────────────
elif MODEL_NAME == 'a2wnet_scratch':
    import keras_hub
    from tensorflow.keras.optimizers import AdamW
    from contributions.hybrid_contrastive import SupervisedContrastiveLoss, multi_loss_generator

    class A2WNet_Scratch(Model):
        """A2WNet_Contrastive with all weights randomly initialized."""
        def __init__(self, num_classes, **kwargs):
            super().__init__(**kwargs)

            # VGG16 branch — no ImageNet weights
            self.vgg_base = VGG16(weights=None, include_top=False,
                                  input_shape=(224, 224, 3))
            self.vgg_pool = layers.GlobalAveragePooling2D()
            self.vgg_proj = layers.Dense(256, activation='relu')

            # ViT-B/16 branch — random init (same architecture, no imagenet21k)
            self.vit_base = keras_hub.models.ViTBackbone(
                image_shape=(224, 224, 3),
                patch_size=16,
                num_layers=12,
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
            )
            self.vit_norm = layers.LayerNormalization(epsilon=1e-6)
            self.vit_proj = layers.Dense(256, activation='gelu')

            # Gating mechanism (same as pretrained version)
            self.concat_for_gate  = layers.Concatenate()
            self.gate_dense       = layers.Dense(256, activation='sigmoid',
                                                 name='attention_gate')
            self.multiply_gate    = layers.Multiply()
            self.gate_inverse     = layers.Lambda(lambda x: 1.0 - x)
            self.multiply_inv_gate = layers.Multiply()
            self.add_features     = layers.Add(name='features')

            self.drop = layers.Dropout(0.3)
            self.out  = layers.Dense(num_classes, activation='softmax',
                                     name='predictions')

        def call(self, inputs, training=False):
            x_vit = inputs / 255.0
            x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))

            v_feat = self.vgg_proj(self.vgg_pool(
                self.vgg_base(x_vgg, training=training)))
            t_feat = self.vit_proj(self.vit_norm(
                self.vit_base(x_vit, training=training)[:, 0, :]))

            gate = self.gate_dense(self.concat_for_gate([v_feat, t_feat]))
            vgg_w = self.multiply_gate([v_feat, gate])
            vit_w = self.multiply_inv_gate([t_feat, self.gate_inverse(gate)])
            blended = self.add_features([vgg_w, vit_w])

            return {
                'predictions': self.out(self.drop(blended, training=training)),
                'features':    blended
            }

    def raw_identity(x): return x
    base_train_gen, base_valid_gen, base_test_gen = make_generators(raw_identity, SEED)

    train_gen = multi_loss_generator(base_train_gen)
    valid_gen = multi_loss_generator(base_valid_gen)

    loss_funcs   = {'predictions': 'categorical_crossentropy',
                    'features':    SupervisedContrastiveLoss(temperature=0.1)}
    loss_weights = {'predictions': 1.0, 'features': 0.15}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    # Train entire network end-to-end from scratch (no freeze phase)
    model = A2WNet_Scratch(num_classes=NUM_CLASSES)
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_funcs, loss_weights=loss_weights,
        metrics={'predictions': ['accuracy']}
    )

    print(f"=== A2WNet-scratch seed={SEED} Phase 1 (lr=1e-3) ===")
    model.fit(
        train_gen, steps_per_epoch=len(base_train_gen),
        validation_data=valid_gen, validation_steps=len(base_valid_gen),
        epochs=15, callbacks=callbacks, verbose=1
    )

    model.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=loss_funcs, loss_weights=loss_weights,
        metrics={'predictions': ['accuracy']}
    )

    print(f"=== A2WNet-scratch seed={SEED} Phase 2 (lr=1e-5) ===")
    model.fit(
        train_gen, steps_per_epoch=len(base_train_gen),
        validation_data=valid_gen, validation_steps=len(base_valid_gen),
        epochs=20, callbacks=callbacks, verbose=1
    )

    test_gen_multi = multi_loss_generator(base_test_gen)
    eval_results  = model.evaluate(
        test_gen_multi, steps=len(base_test_gen), verbose=0
    )
    metrics_dict = dict(zip(model.metrics_names, eval_results))
    test_loss = next((float(v) for k, v in metrics_dict.items()
                      if 'predictions' in k and 'loss' in k), float(eval_results[0]))
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
