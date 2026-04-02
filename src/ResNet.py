import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
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

dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
df = load_dataset(dataset_dir)

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

train_gen_ResNet, valid_gen_ResNet, test_gen_ResNet = gen(preprocess_input, train_df, test_df)

NUM_CLASSES = len(df['label'].unique())


# ── Model definition ───────────────────────────────────────────────────────────

def create_resnet_model(num_classes, freeze_base=True):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


# ── Callbacks ──────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, mode='min',
                  restore_best_weights=True),
    ModelCheckpoint(filepath='best_resnet.weights.h5', monitor='val_loss',
                    save_best_only=True, mode='min', save_weights_only=True)
]


# ── Phase 1: train head only ───────────────────────────────────────────────────

model_resnet, base_model = create_resnet_model(num_classes=NUM_CLASSES, freeze_base=True)

model_resnet.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Phase 1: Training classification head ===")
history_phase1 = model_resnet.fit(
    train_gen_ResNet,
    validation_data=valid_gen_ResNet,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)


# ── Phase 2: fine-tune conv5_block onwards ──────────────────────────────────

base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    # ResNet50 has layers named conv5_block1_1_conv, etc.
    if 'conv5_block' in layer.name:
        set_trainable = True
    layer.trainable = set_trainable

model_resnet.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Phase 2: Fine-tuning from conv5_block onwards ===")
history_phase2 = model_resnet.fit(
    train_gen_ResNet,
    validation_data=valid_gen_ResNet,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)


# ── Evaluation ─────────────────────────────────────────────────────────────────

plot_history(history_phase2, test_gen_ResNet, train_gen_ResNet, model_resnet, test_df)

result_ResNet = result_test(test_gen_ResNet, model_resnet)
