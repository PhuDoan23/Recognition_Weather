import os
import pandas as pd
import tensorflow as tf

# Limit threads to avoid EAGAIN (Check failed: ret == 0 (11 vs. 0) Thread tf_ creation via pthread_create() failed)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'

import keras_hub
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from utils import gen, plot_history, result_test


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


PRESET = 'vit_base_patch16_224_imagenet21k'


class HybridGatedModel(Model):
    def __init__(self, num_classes, freeze_base=True, **kwargs):
        super().__init__(**kwargs)

        # Branch 1: VGG16 (Local)
        self.vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.vgg_pool = layers.GlobalAveragePooling2D()
        self.vgg_proj = layers.Dense(256, activation='relu', name='vgg_proj')

        # Branch 2: ViT (Global)
        self.vit_base = keras_hub.models.ViTBackbone.from_preset(PRESET)
        self.vit_norm = layers.LayerNormalization(epsilon=1e-6)
        self.vit_proj = layers.Dense(256, activation='gelu', name='vit_proj')

        self.vgg_base.trainable = not freeze_base
        self.vit_base.trainable = not freeze_base

        # --- DYNAMIC GATING MECHANISM ---
        # The neural network will learn a 256-dimensional "gate vector" with values between 0 and 1.
        # This gate decides per-feature whether to trust the Local CNN (1) or Global ViT (0).
        self.concat_for_gate = layers.Concatenate()
        self.gate_dense = layers.Dense(256, activation='sigmoid', name='attention_gate')

        self.multiply_gate = layers.Multiply(name='vgg_gated')             # vgg * gate
        self.gate_inverse = layers.Lambda(lambda x: 1.0 - x, name='invert_gate')
        self.multiply_inv_gate = layers.Multiply(name='vit_gated')         # vit * (1 - gate)

        self.add_features = layers.Add() # Computes: (vgg * gate) + (vit * (1 - gate))
        # --------------------------------

        # Final classification head
        self.drop = layers.Dropout(0.3)
        self.out = layers.Dense(num_classes, activation='softmax', name='predictions')

        self.num_classes = num_classes
        self.freeze_base = freeze_base

    def call(self, inputs, training=False):
        # 1. Image preprocessing inside the model
        x_vit = inputs / 255.0
        x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))

        # 2. Extract Features
        v_feat = self.vgg_base(x_vgg, training=False)
        v_feat = self.vgg_pool(v_feat)
        v_feat = self.vgg_proj(v_feat)

        t_feat = self.vit_base(x_vit, training=False)[:, 0, :]
        t_feat = self.vit_norm(t_feat)
        t_feat = self.vit_proj(t_feat)

        # 3. Dynamic Gating Fusion
        # Concatenate features to provide full context to the gate network
        concat_context = self.concat_for_gate([v_feat, t_feat])

        # Generate the [0, 1] weighting vector
        gate = self.gate_dense(concat_context)
        inv_gate = self.gate_inverse(gate)

        # Apply the adaptive weights
        vgg_weighted = self.multiply_gate([v_feat, gate])
        vit_weighted = self.multiply_inv_gate([t_feat, inv_gate])

        # Blend the features adaptively
        blended_features = self.add_features([vgg_weighted, vit_weighted])

        # 4. Classification Head
        out = self.drop(blended_features, training=training)
        return self.out(out)

    def get_embeddings(self, inputs):
        """Return 256-dim blended features before the classifier (for t-SNE)."""
        x_vit = inputs / 255.0
        x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))
        v_feat = self.vgg_proj(self.vgg_pool(self.vgg_base(x_vgg, training=False)))
        t_feat = self.vit_proj(self.vit_norm(self.vit_base(x_vit, training=False)[:, 0, :]))
        gate          = self.gate_dense(self.concat_for_gate([v_feat, t_feat]))
        vgg_weighted  = self.multiply_gate([v_feat, gate])
        vit_weighted  = self.multiply_inv_gate([t_feat, self.gate_inverse(gate)])
        return self.add_features([vgg_weighted, vit_weighted])

    def unfreeze_last_blocks(self, vit_blocks=4, vgg_blocks=2):
        self.vgg_base.trainable = True
        self.vit_base.trainable = True

        # Unfreeze Top VGG blocks
        freeze_vgg_until = -(vgg_blocks * 4)
        for layer in self.vgg_base.layers[:freeze_vgg_until]:
            layer.trainable = False

        # Unfreeze Top ViT transformer blocks
        transformer_layers = [l for l in self.vit_base.layers if 'transformer' in l.name.lower()]
        freeze_vit_until = len(transformer_layers) - vit_blocks
        for i, layer in enumerate(transformer_layers):
            layer.trainable = i >= freeze_vit_until


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    df = load_dataset(dataset_dir)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    NUM_CLASSES = len(df['label'].unique())

    # Pass None for pre so the generator returns raw [0, 255] RGB float tensors.
    def raw_identity(x):
        return x
    train_gen_Hybrid, valid_gen_Hybrid, test_gen_Hybrid = gen(raw_identity, train_df, test_df)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True),
        ModelCheckpoint(filepath='models/best_hybrid_gated.weights.h5', monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    os.makedirs('models', exist_ok=True)

    # Phase 1: Train just the heads and the gate
    model_gated = HybridGatedModel(num_classes=NUM_CLASSES, freeze_base=True)
    model_gated.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== Phase 1: Training Gated Fusion Head ===")
    history_phase1 = model_gated.fit(
        train_gen_Hybrid,
        validation_data=valid_gen_Hybrid,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune the tops of both bases
    model_gated.unfreeze_last_blocks(vit_blocks=4, vgg_blocks=2)
    model_gated.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== Phase 2: Fine-tuning Bases alongside Gate ===")
    history_phase2 = model_gated.fit(
        train_gen_Hybrid,
        validation_data=valid_gen_Hybrid,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs('Img', exist_ok=True)
    plot_history(history_phase2, test_gen_Hybrid, train_gen_Hybrid, model_gated, test_df)
    result_Hybrid = result_test(test_gen_Hybrid, model_gated)
