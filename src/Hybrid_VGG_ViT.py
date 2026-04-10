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

# Ensure local utils imports work
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

dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
df = load_dataset(dataset_dir)

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
NUM_CLASSES = len(df['label'].unique())

# Pass None for pre so the generator returns raw [0, 255] RGB float tensors.
# Preprocessing is handled inside the Model graph for each branch.
def raw_identity(x):
    return x

train_gen_Hybrid, valid_gen_Hybrid, test_gen_Hybrid = gen(raw_identity, train_df, test_df)

PRESET = 'vit_base_patch16_224_imagenet21k'
DROPOUT_RATE = 0.3

class HybridVGGViT(Model):
    def __init__(self, num_classes, freeze_base=True, **kwargs):
        super().__init__(**kwargs)
        
        # Branch 1: VGG16
        self.vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.vgg_pool = layers.GlobalAveragePooling2D()
        self.vgg_proj = layers.Dense(256, activation='relu', name='vgg_proj')
        
        # Branch 2: ViT
        self.vit_base = keras_hub.models.ViTBackbone.from_preset(PRESET)
        self.vit_norm = layers.LayerNormalization(epsilon=1e-6)
        self.vit_proj = layers.Dense(256, activation='gelu', name='vit_proj')
        
        self.vgg_base.trainable = not freeze_base
        self.vit_base.trainable = not freeze_base
        
        # Fusion head
        self.concat = layers.Concatenate()
        self.fusion_dense = layers.Dense(256, activation='gelu', name='fusion_dense')
        self.drop = layers.Dropout(DROPOUT_RATE)
        self.out = layers.Dense(num_classes, activation='softmax', name='predictions')
        
        self.num_classes = num_classes
        self.freeze_base = freeze_base

    def call(self, inputs, training=False):
        # 1. Image preprocessing inside the model
        
        # ViT typically expects [0, 1] scaled images
        x_vit = inputs / 255.0
        
        # VGG expects Caffe format (BGR, zero-centered around ImageNet means channel-wise)
        # Using tf.cast to ensure standard float32 precision
        x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))
        
        # 2. VGG forward pass
        v_feat = self.vgg_base(x_vgg, training=False)
        v_feat = self.vgg_pool(v_feat)
        v_feat = self.vgg_proj(v_feat)
        
        # 3. ViT forward pass (sequence output, index 0 is CLS token)
        t_feat = self.vit_base(x_vit, training=False)[:, 0, :]
        t_feat = self.vit_norm(t_feat)
        t_feat = self.vit_proj(t_feat)
        
        # 4. Fusion and Classification Head
        fused = self.concat([v_feat, t_feat])
        fused = self.fusion_dense(fused)
        fused = self.drop(fused, training=training)
        
        return self.out(fused)

    def get_embeddings(self, inputs):
        """Return 256-dim fused features before the classifier (for t-SNE)."""
        x_vit = tf.cast(inputs, tf.float32) / 255.0
        x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))
        v_feat = self.vgg_proj(self.vgg_pool(self.vgg_base(x_vgg, training=False)))
        t_feat = self.vit_proj(self.vit_norm(self.vit_base(x_vit, training=False)[:, 0, :]))
        fused  = self.concat([v_feat, t_feat])
        return self.fusion_dense(fused)

    def unfreeze_last_blocks(self, vit_blocks=4, vgg_blocks=2):
        self.vgg_base.trainable = True
        self.vit_base.trainable = True
        
        # Unfreeze Top VGG blocks (e.g. block5)
        # 1 block = 4 layers usually: 3 conv + 1 pool. VGG ending is -4 to -8 layers.
        freeze_vgg_until = -(vgg_blocks * 4)
        for layer in self.vgg_base.layers[:freeze_vgg_until]:
            layer.trainable = False
            
        # Unfreeze Top ViT transformer blocks
        transformer_layers = [l for l in self.vit_base.layers if 'transformer' in l.name.lower()]
        freeze_vit_until = len(transformer_layers) - vit_blocks
        for i, layer in enumerate(transformer_layers):
            layer.trainable = i >= freeze_vit_until

if __name__ == '__main__':
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True),
        ModelCheckpoint(filepath='models/best_hybrid_vgg_vit.weights.h5', monitor='val_loss',
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Phase 1: Train just the heads
    model_hybrid = HybridVGGViT(num_classes=NUM_CLASSES, freeze_base=True)
    model_hybrid.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== Phase 1: Training fusion head (bases frozen) ===")
    history_phase1 = model_hybrid.fit(
        train_gen_Hybrid,
        validation_data=valid_gen_Hybrid,
        epochs=15, # Shorter epochs
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune the tops of both bases
    model_hybrid.unfreeze_last_blocks(vit_blocks=4, vgg_blocks=2)
    model_hybrid.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("=== Phase 2: Fine-tuning last blocks of VGG and ViT ===")
    history_phase2 = model_hybrid.fit(
        train_gen_Hybrid,
        validation_data=valid_gen_Hybrid,
        epochs=20, # Shorter epochs
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs('Img', exist_ok=True)
    plot_history(history_phase2, test_gen_Hybrid, train_gen_Hybrid, model_hybrid, test_df)
    result_Hybrid = result_test(test_gen_Hybrid, model_hybrid)
