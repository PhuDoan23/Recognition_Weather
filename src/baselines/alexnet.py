import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import preprocess_input
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

train_gen_Alex, valid_gen_Alex, test_gen_Alex = gen(preprocess_input, train_df, test_df)

NUM_CLASSES = len(df['label'].unique())


# ── Model definition ───────────────────────────────────────────────────────────
#
# Modernized AlexNet for 224x224 input:
#   Original AlexNet used 227x227 with stride-4 on Conv1.
#   At 224x224 with stride-4 the spatial dims work out cleanly:
#     Conv1 (11x11, s=4) → 54x54  → MaxPool (3x3, s=2) → 26x26
#     Conv2 (5x5,  s=1) → 26x26  → MaxPool (3x3, s=2) → 12x12
#     Conv3 (3x3,  s=1) → 12x12
#     Conv4 (3x3,  s=1) → 12x12
#     Conv5 (3x3,  s=1) → 12x12  → MaxPool (3x3, s=2) →  5x5
#     Flatten → 5*5*256 = 6400
#   BatchNormalization added after each conv block (not in original).

class AlexNetClassifier(Model):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()

        # Block 1
        self.conv1 = Conv2D(96, (11, 11), strides=4, activation='relu',
                            padding='valid')
        self.bn1   = BatchNormalization()
        self.pool1 = MaxPooling2D((3, 3), strides=2)

        # Block 2
        self.conv2 = Conv2D(256, (5, 5), strides=1, activation='relu',
                            padding='same')
        self.bn2   = BatchNormalization()
        self.pool2 = MaxPooling2D((3, 3), strides=2)

        # Block 3
        self.conv3 = Conv2D(384, (3, 3), strides=1, activation='relu',
                            padding='same')
        self.bn3   = BatchNormalization()

        # Block 4
        self.conv4 = Conv2D(384, (3, 3), strides=1, activation='relu',
                            padding='same')
        self.bn4   = BatchNormalization()

        # Block 5
        self.conv5 = Conv2D(256, (3, 3), strides=1, activation='relu',
                            padding='same')
        self.bn5   = BatchNormalization()
        self.pool5 = MaxPooling2D((3, 3), strides=2)

        # Classifier head
        self.flatten = Flatten()
        self.dense1  = Dense(4096, activation='relu')
        self.drop1   = Dropout(dropout_rate)
        self.dense2  = Dense(4096, activation='relu')
        self.drop2   = Dropout(dropout_rate)
        self.out     = Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x, training=training)

        # Block 5
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.pool5(x)

        # Head
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop1(x, training=training)
        x = self.dense2(x)
        x = self.drop2(x, training=training)
        return self.out(x)


# ── Callbacks ──────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, mode='min',
                  restore_best_weights=True),
    ModelCheckpoint(filepath='best_alexnet.weights.h5', monitor='val_loss',
                    save_best_only=True, mode='min', save_weights_only=True)
]


# ── Training ───────────────────────────────────────────────────────────────────

model_alexnet = AlexNetClassifier(num_classes=NUM_CLASSES)
model_alexnet.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== AlexNet: Training ===")
history_alexnet = model_alexnet.fit(
    train_gen_Alex,
    validation_data=valid_gen_Alex,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)


# ── Evaluation ─────────────────────────────────────────────────────────────────

plot_history(history_alexnet, test_gen_Alex, train_gen_Alex, model_alexnet, test_df)
result_AlexNet = result_test(test_gen_Alex, model_alexnet)
