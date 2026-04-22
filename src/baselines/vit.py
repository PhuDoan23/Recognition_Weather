import os
import pandas as pd
import tensorflow as tf
import keras_hub
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import plot_history, result_test


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(dataset_dir):
    records = []
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({
                    'filename': os.path.join(dataset_dir, label, fname),
                    'label': label
                })
    return pd.DataFrame(records)


# ── Model definition ───────────────────────────────────────────────────────────
#
# Pretrained base: ViT-B/16 from ImageNet-21k (21k-class pretraining → richer features)
#   - 12 Transformer blocks, hidden_dim=768, 12 heads, patch_size=16
#   - Input: 224x224, output: (B, 197, 768) sequence; index 0 = CLS token
#
# Custom head:
#   CLS token → LayerNorm → Dense(256, gelu) → Dropout → Dense(11, softmax)
#
# Two-phase training (same strategy as VGG16/ResNet):
#   Phase 1 — freeze base, train head only
#   Phase 2 — unfreeze last 4 transformer blocks, fine-tune at low lr

PRESET = 'vit_base_patch16_224_imagenet21k'
DROPOUT_RATE = 0.3


class ViTClassifier(Model):
    def __init__(self, num_classes, dropout_rate=DROPOUT_RATE, freeze_base=True, **kwargs):
        super().__init__(**kwargs)

        self.backbone = keras_hub.models.ViTBackbone.from_preset(PRESET)
        self.backbone.trainable = not freeze_base

        self.norm    = layers.LayerNormalization(epsilon=1e-6)
        self.dense1  = layers.Dense(256, activation='gelu')
        self.drop    = layers.Dropout(dropout_rate)
        self.out     = layers.Dense(num_classes, activation='softmax')

        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_base  = freeze_base

    def call(self, x, training=False):
        # backbone output shape: (B, 197, 768) — index 0 is the CLS token
        x = self.backbone(x, training=False)
        cls = x[:, 0, :]                      # (B, 768)
        cls = self.norm(cls)
        cls = self.dense1(cls)
        cls = self.drop(cls, training=training)
        return self.out(cls)

    def unfreeze_last_blocks(self, num_blocks=4):
        """Unfreeze the last N transformer blocks for fine-tuning."""
        self.backbone.trainable = True
        # ViT-B/16 has 12 transformer blocks; freeze all except the last num_blocks
        transformer_layers = [l for l in self.backbone.layers
                               if 'transformer' in l.name.lower()]
        freeze_until = len(transformer_layers) - num_blocks
        for i, layer in enumerate(transformer_layers):
            layer.trainable = i >= freeze_until

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'freeze_base': self.freeze_base
        })
        return config


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    df = load_dataset(dataset_dir)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    NUM_CLASSES = len(df['label'].unique())

    # ViT-B/16 from ImageNet expects inputs scaled to [0, 1]
    def rescale(x):
        return x / 255.0

    train_datagen = ImageDataGenerator(
        preprocessing_function=rescale,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(preprocessing_function=rescale)

    train_gen_ViT = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filename', y_col='label',
        target_size=(224, 224), class_mode='categorical',
        batch_size=32, shuffle=True, seed=0, subset='training'
    )
    valid_gen_ViT = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filename', y_col='label',
        target_size=(224, 224), class_mode='categorical',
        batch_size=32, shuffle=False, seed=0, subset='validation'
    )
    test_gen_ViT = test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='filename', y_col='label',
        target_size=(224, 224), color_mode='rgb',
        class_mode='categorical', batch_size=32, shuffle=False, verbose=0
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath='best_vit.weights.h5', monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    # ── Phase 1: train head only, backbone frozen ──────────────────────────────

    model_vit = ViTClassifier(num_classes=NUM_CLASSES, freeze_base=True)
    model_vit.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== ViT Phase 1: Training classification head ===")
    history_phase1 = model_vit.fit(
        train_gen_ViT,
        validation_data=valid_gen_ViT,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: unfreeze last 4 transformer blocks ────────────────────────────

    model_vit.unfreeze_last_blocks(num_blocks=4)
    model_vit.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== ViT Phase 2: Fine-tuning last 4 transformer blocks ===")
    history_phase2 = model_vit.fit(
        train_gen_ViT,
        validation_data=valid_gen_ViT,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluation ────────────────────────────────────────────────────────────

    plot_history(history_phase2, test_gen_ViT, train_gen_ViT, model_vit, test_df)
    result_ViT = result_test(test_gen_ViT, model_vit)
