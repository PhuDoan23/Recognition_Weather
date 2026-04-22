import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
import keras_hub

# Limit threads to avoid warnings
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'

PRESET = 'vit_base_patch16_224_imagenet21k'
NUM_CLASSES = 11

class A2WNet_Contrastive(Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.vgg_pool = layers.GlobalAveragePooling2D()
        self.vgg_proj = layers.Dense(256, activation='relu')
        self.vit_base = keras_hub.models.ViTBackbone.from_preset(PRESET)
        self.vit_norm = layers.LayerNormalization(epsilon=1e-6)
        self.vit_proj = layers.Dense(256, activation='gelu')
        self.concat_for_gate = layers.Concatenate()
        self.gate_dense = layers.Dense(256, activation='sigmoid')
        self.multiply_gate = layers.Multiply()
        self.gate_inverse = layers.Lambda(lambda x: 1.0 - x)
        self.multiply_inv_gate = layers.Multiply()
        self.add_features = layers.Add(name='features')
        self.drop = layers.Dropout(0.3)
        self.out = layers.Dense(num_classes, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x_vit = inputs / 255.0
        x_vgg = vgg_preprocess(tf.cast(inputs, tf.float32))
        v_feat = self.vgg_proj(self.vgg_pool(self.vgg_base(x_vgg, training=False)))
        t_feat = self.vit_proj(self.vit_norm(self.vit_base(x_vit, training=False)[:, 0, :]))
        gate = self.gate_dense(self.concat_for_gate([v_feat, t_feat]))
        vgg_weighted = self.multiply_gate([v_feat, gate])
        vit_weighted = self.multiply_inv_gate([t_feat, self.gate_inverse(gate)])
        blended = self.add_features([vgg_weighted, vit_weighted])
        return {'predictions': self.out(self.drop(blended, training=training)), 'features': blended}

class VGG16Classifier(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.pool = layers.GlobalAveragePooling2D()
        self.bn1 = layers.BatchNormalization()
        self.dense1 = layers.Dense(512, activation='relu')
        self.drop1 = layers.Dropout(0.5)
        self.bn2 = layers.BatchNormalization()
        self.dense2 = layers.Dense(256, activation='relu')
        self.drop2 = layers.Dropout(0.5)
        self.out = layers.Dense(num_classes, activation='softmax')
        
    def call(self, x, training=False):
        x = self.base(x, training=False)
        x = self.pool(x)
        x = self.drop1(self.dense1(self.bn1(x, training=training)), training=training)
        x = self.drop2(self.dense2(self.bn2(x, training=training)), training=training)
        return self.out(x)

def get_img_array(img_path, size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_axes(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, full_model, model_type="vgg"):
    """
    Computes the Grad-CAM heatmap for a given model.
    """
    # We need to create a sub-model that outputs both the target conv layer's output and the final prediction.
    if model_type == "vgg":
        last_conv_layer = full_model.base.get_layer("block5_conv3")
        # Sub-model mapping input -> (last_conv_output)
        conv_model = Model(full_model.base.inputs, last_conv_layer.output)
        
        with tf.GradientTape() as tape:
            # Forward pass
            img_vgg = vgg_preprocess(tf.cast(img_array, tf.float32))
            conv_outputs = conv_model(img_vgg)
            tape.watch(conv_outputs)
            
            # Continue the forward pass using the remaining layers of VGG16Classifier
            x = full_model.pool(conv_outputs)
            x = full_model.bn1(x, training=False)
            x = full_model.dense1(x)
            x = full_model.drop1(x, training=False)
            x = full_model.bn2(x, training=False)
            x = full_model.dense2(x)
            x = full_model.drop2(x, training=False)
            preds = full_model.out(x)
            
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
            
        grads = tape.gradient(top_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), int(top_pred_index)
        
    elif model_type == "a2wnet":
        last_conv_layer = full_model.vgg_base.get_layer("block5_conv3")
        conv_model = Model(full_model.vgg_base.inputs, last_conv_layer.output)
        
        with tf.GradientTape() as tape:
            x_vit = img_array / 255.0
            x_vgg = vgg_preprocess(tf.cast(img_array, tf.float32))
            
            conv_outputs = conv_model(x_vgg)
            tape.watch(conv_outputs)
            
            # VGG branch remaining
            v_feat = full_model.vgg_proj(full_model.vgg_pool(conv_outputs))
            
            # ViT branch
            t_feat = full_model.vit_proj(full_model.vit_norm(full_model.vit_base(x_vit, training=False)[:, 0, :]))
            
            # Fusion
            gate = full_model.gate_dense(full_model.concat_for_gate([v_feat, t_feat]))
            vgg_weighted = full_model.multiply_gate([v_feat, gate])
            vit_weighted = full_model.multiply_inv_gate([t_feat, full_model.gate_inverse(gate)])
            blended = full_model.add_features([vgg_weighted, vit_weighted])
            preds = full_model.out(full_model.drop(blended, training=False))
            
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
            
        grads = tape.gradient(top_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), int(top_pred_index)

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return img, superimposed_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--output', type=str, default='gradcam_result.png', help='Output image path')
    args = parser.parse_args()
    
    print("Loading models...")
    vgg_model = VGG16Classifier(NUM_CLASSES)
    a2wnet_model = A2WNet_Contrastive(NUM_CLASSES)
    
    dummy_input = tf.zeros((1, 224, 224, 3))
    vgg_model(dummy_input)
    a2wnet_model(dummy_input)
    
    vgg_weights = '../models/best_vgg16.weights.h5'
    a2wnet_weights = '../models/best_A2WNet_Contrastive.weights.h5'
    
    vgg_model.load_weights(os.path.join(os.path.dirname(__file__), vgg_weights))
    a2wnet_model.load_weights(os.path.join(os.path.dirname(__file__), a2wnet_weights))
    
    img_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.image, target_size=(224, 224))
    ), axis=0)
    
    print("Computing VGG16 Grad-CAM...")
    heatmap_vgg, pred_vgg = make_gradcam_heatmap(img_array, vgg_model, "vgg")
    
    print("Computing A2WNet_Contrastive Grad-CAM...")
    heatmap_a2w, pred_a2w = make_gradcam_heatmap(img_array, a2wnet_model, "a2wnet")
    
    # Class names mapping from notebook
    class_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
    
    orig_img, super_vgg = display_gradcam(args.image, heatmap_vgg)
    _, super_a2w = display_gradcam(args.image, heatmap_a2w)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(super_vgg)
    plt.title(f"VGG16 (Pred: {class_names[pred_vgg]})")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(super_a2w)
    plt.title(f"A2WNet (Pred: {class_names[pred_a2w]})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    pdf_output = os.path.splitext(args.output)[0] + '.pdf'
    plt.savefig(pdf_output, bbox_inches='tight')
    print(f"Saved visualization to {args.output} and {pdf_output}")

if __name__ == '__main__':
    main()
