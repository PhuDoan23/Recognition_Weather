import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import gen, plot_history, result_test


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

class VGG16Classifier(Model):
    def __init__(self, num_classes, dropout_rate=0.5, freeze_base=True):
        super().__init__()

        self.base = VGG16(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
        self.base.trainable = not freeze_base

        self.pool   = GlobalAveragePooling2D()
        self.bn1    = BatchNormalization()
        self.dense1 = Dense(512, activation='relu')
        self.drop1  = Dropout(dropout_rate)
        self.bn2    = BatchNormalization()
        self.dense2 = Dense(256, activation='relu')
        self.drop2  = Dropout(dropout_rate)
        self.out    = Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.base(x, training=False)
        x = self.pool(x)
        x = self.bn1(x, training=training)
        x = self.dense1(x)
        x = self.drop1(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dense2(x)
        x = self.drop2(x, training=training)
        return self.out(x)

    def unfreeze_top_blocks(self, num_blocks=2):
        self.base.trainable = True
        freeze_until = -(num_blocks * 4)
        for layer in self.base.layers[:freeze_until]:
            layer.trainable = False


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    df = load_dataset(dataset_dir)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    train_gen_VGG, valid_gen_VGG, test_gen_VGG = gen(preprocess_input, train_df, test_df)

    NUM_CLASSES = len(df['label'].unique())

    # ── Callbacks ─────────────────────────────────────────────────────────────

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, mode='min',
                      restore_best_weights=True),
        ModelCheckpoint(filepath='best_vgg16.weights.h5', monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    # ── Phase 1: train head only ───────────────────────────────────────────────

    model_vgg = VGG16Classifier(num_classes=NUM_CLASSES, freeze_base=True)
    model_vgg.compile(
        optimizer=Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== Phase 1: Training classification head ===")
    history_phase1 = model_vgg.fit(
        train_gen_VGG,
        validation_data=valid_gen_VGG,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: fine-tune last 2 conv blocks ─────────────────────────────────

    model_vgg.unfreeze_top_blocks(num_blocks=2)
    model_vgg.compile(
        optimizer=Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== Phase 2: Fine-tuning top conv blocks ===")
    history_phase2 = model_vgg.fit(
        train_gen_VGG,
        validation_data=valid_gen_VGG,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluation ────────────────────────────────────────────────────────────

    plot_history(history_phase2, test_gen_VGG, train_gen_VGG, model_vgg, test_df)
    result_VGG16 = result_test(test_gen_VGG, model_vgg)
