import os
import pandas as pd
import tensorflow as tf

# Limit threads to avoid EAGAIN (Check failed: ret == 0 (11 vs. 0) Thread tf_ creation via pthread_create() failed)
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'

import keras_hub
import numpy as np
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


# Wrapper to yield dual targets: one for classification (CCE), one for embeddings (SupCon)
def multi_loss_generator(base_gen):
    while True:
        x, y = next(base_gen)
        yield x, {'predictions': y, 'features': y}


# ── Mathematical Contribution: Supervised Contrastive Loss ───────────────────
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name='supcon_loss'):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, labels, feature_vectors):
        # Normalize the embedding features onto the unit hypersphere
        feature_vectors = tf.math.l2_normalize(feature_vectors, axis=1)

        # Compute Cosine Similarity scaled by temperature
        logits = tf.divide(tf.matmul(feature_vectors, tf.transpose(feature_vectors)), self.temperature)

        # Convert categorical one-hot labels to flat indices
        labels = tf.argmax(labels, axis=1)
        labels = tf.cast(labels, tf.int32)
        labels = tf.expand_dims(labels, -1)

        # Create a mask where mask[i, j] is 1 if sample i and j have the SAME class
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        # Numerical stability tricks for softmax implementation
        logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
        logits = logits - tf.stop_gradient(logits_max)

        # Remove the main diagonal (a sample's similarity to itself) from the mask
        batch_size  = tf.shape(logits)[0]
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
        mask = mask * logits_mask

        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)

        # Compute the mean log-likelihood over the positive pairs in the batch
        sum_mask = tf.reduce_sum(mask, axis=1)
        sum_mask = tf.maximum(sum_mask, 1.0) # avoid division by zero if class appears only once in batch
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / sum_mask

        loss = -mean_log_prob_pos
        return tf.reduce_mean(loss)


# ── The Architecture: A2W-Net (Adaptive Attention Weather Network) ─────────────
PRESET = 'vit_base_patch16_224_imagenet21k'

class A2WNet_Contrastive(Model):
    def __init__(self, num_classes, freeze_base=True, **kwargs):
        super().__init__(**kwargs)

        # Local CNN Branch
        self.vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.vgg_pool = layers.GlobalAveragePooling2D()
        self.vgg_proj = layers.Dense(256, activation='relu')

        # Global Transformer Branch
        self.vit_base = keras_hub.models.ViTBackbone.from_preset(PRESET)
        self.vit_norm = layers.LayerNormalization(epsilon=1e-6)
        self.vit_proj = layers.Dense(256, activation='gelu')

        self.vgg_base.trainable = not freeze_base
        self.vit_base.trainable = not freeze_base

        # Gating Mechanism (Adaptive Weights)
        self.concat_for_gate = layers.Concatenate()
        self.gate_dense = layers.Dense(256, activation='sigmoid', name='attention_gate')

        self.multiply_gate = layers.Multiply()
        self.gate_inverse = layers.Lambda(lambda x: 1.0 - x)
        self.multiply_inv_gate = layers.Multiply()
        self.add_features = layers.Add(name='features') # The embedding layer we will plot in t-SNE

        # Classifier Head
        self.drop = layers.Dropout(0.3)
        self.out = layers.Dense(num_classes, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        # 1. Preprocessing
        x_vit = inputs / 255.0
        x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))

        # 2. Extract Streams
        v_feat = self.vgg_proj(self.vgg_pool(self.vgg_base(x_vgg, training=False)))
        t_feat = self.vit_proj(self.vit_norm(self.vit_base(x_vit, training=False)[:, 0, :]))

        # 3. Dynamic Gating
        gate = self.gate_dense(self.concat_for_gate([v_feat, t_feat]))
        vgg_weighted = self.multiply_gate([v_feat, gate])
        vit_weighted = self.multiply_inv_gate([t_feat, self.gate_inverse(gate)])

        # This blended feature vector is the "latent space embedding" we pass to SupCon Loss
        blended_features = self.add_features([vgg_weighted, vit_weighted])

        # 4. Classification Output
        predicted_probs = self.out(self.drop(blended_features, training=training))

        # We output a dictionary matching the multi_loss_generator target keys
        return {'predictions': predicted_probs, 'features': blended_features}

    def unfreeze_last_blocks(self, vit_blocks=4, vgg_blocks=2):
        self.vgg_base.trainable = True
        self.vit_base.trainable = True

        freeze_vgg_until = -(vgg_blocks * 4)
        for layer in self.vgg_base.layers[:freeze_vgg_until]:
            layer.trainable = False

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

    def raw_identity(x): return x
    train_gen_base, valid_gen_base, test_gen_base = gen(raw_identity, train_df, test_df)

    train_gen_Hybrid = multi_loss_generator(train_gen_base)
    valid_gen_Hybrid = multi_loss_generator(valid_gen_base)
    test_gen_Hybrid  = multi_loss_generator(test_gen_base)

    os.makedirs('models', exist_ok=True)
    os.makedirs('Img', exist_ok=True)

    # We heavily weigh the normal CCE loss, but add a 0.2 weight for Contrastive Loss
    # to explicitly organize the latent space geometry.
    loss_funcs = {
        'predictions': 'categorical_crossentropy',
        'features': SupervisedContrastiveLoss(temperature=0.1)
    }
    loss_weights = {'predictions': 1.0, 'features': 0.15}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True),
        ModelCheckpoint(filepath='models/best_A2WNet_Contrastive.weights.h5', monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    model = A2WNet_Contrastive(num_classes=NUM_CLASSES, freeze_base=True)
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_funcs,
        loss_weights=loss_weights,
        metrics={'predictions': ['accuracy']}
    )

    print("=== Phase 1: Training A2W-Net Fusion Gate with Supervised Contrastive Loss ===")
    history_phase1 = model.fit(
        train_gen_Hybrid,
        steps_per_epoch=len(train_gen_base),
        validation_data=valid_gen_Hybrid,
        validation_steps=len(valid_gen_base),
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    model.unfreeze_last_blocks()
    model.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=loss_funcs,
        loss_weights=loss_weights,
        metrics={'predictions': ['accuracy']}
    )

    print("=== Phase 2: Fine-Tuning Full Architecture ===")
    history_phase2 = model.fit(
        train_gen_Hybrid,
        steps_per_epoch=len(train_gen_base),
        validation_data=valid_gen_Hybrid,
        validation_steps=len(valid_gen_base),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    print("Training Complete! The weights are saved to models/best_A2WNet_Contrastive.weights.h5")
    print("Proceed to run plot_tsne.py to visualize the Contrastive separation!")
