import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout
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

dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
df = load_dataset(dataset_dir)

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

train_gen_MobileNet, valid_gen_MobileNet, test_gen_MobileNet = gen(preprocess_input, train_df, test_df)

NUM_CLASSES = len(df['label'].unique())


# ── Model definition ───────────────────────────────────────────────────────────

def create_mobilenet_model(num_classes, freeze_base=True):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        alpha=1.0
    )
    
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


# ── Callbacks ──────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, mode='min',
                  restore_best_weights=True),
    ModelCheckpoint(filepath='best_mobilenet.weights.h5', monitor='val_loss',
                    save_best_only=True, mode='min', save_weights_only=True)
]


# ── Phase 1: train head only ───────────────────────────────────────────────────

model_mobilenet, base_model = create_mobilenet_model(num_classes=NUM_CLASSES, freeze_base=True)

model_mobilenet.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Phase 1: Training classification head ===")
history_phase1 = model_mobilenet.fit(
    train_gen_MobileNet,
    validation_data=valid_gen_MobileNet,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)


# ── Phase 2: fine-tune from block 16 onwards ──────────────────────────────────

base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    if layer.name.startswith('block_16'):
        set_trainable = True
    layer.trainable = set_trainable

model_mobilenet.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Phase 2: Fine-tuning from block 16 onwards ===")
history_phase2 = model_mobilenet.fit(
    train_gen_MobileNet,
    validation_data=valid_gen_MobileNet,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)


# ── Evaluation ─────────────────────────────────────────────────────────────────

# Note: plot_history in utils.py seems to have some global variable dependencies
# but we follow the call pattern from VGG16.py
plot_history(history_phase2, test_gen_MobileNet, train_gen_MobileNet, model_mobilenet, test_df)

result_MobileNet = result_test(test_gen_MobileNet, model_mobilenet)
