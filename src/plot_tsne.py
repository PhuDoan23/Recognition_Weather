"""
t-SNE Comparison: Hard Concat  vs  Gated  vs  A2WNet Contrastive
=================================================================
Produces two figures saved to figures/:

  Fig 1 — tsne_comparison.png
    3-panel side-by-side t-SNE of all 11 classes, one panel per model.
    Shows how the latent space structure evolves across the three models.

  Fig 2 — tsne_icy_cluster.png
    3-panel zoom on the icy ambiguity cluster (frost / glaze / rime / snow / dew).
    All other classes are grayed out.
    This is the key scientific claim: contrastive loss separates the icy cluster.

Usage (run from project root):
    python src/plot_tsne.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contributions.hybrid_vgg_vit    import HybridVGGViT,         load_dataset, raw_identity
from contributions.hybrid_gated      import HybridGatedModel
from contributions.hybrid_contrastive import A2WNet_Contrastive
from utils import gen

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
IMG_DIR      = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(IMG_DIR, exist_ok=True)

WEIGHTS = {
    'Hard Concat' : os.path.join(MODELS_DIR, 'best_hybrid_vgg_vit.weights.h5'),
    'Gated'       : os.path.join(MODELS_DIR, 'best_hybrid_gated.weights.h5'),
    'A2WNet\n(Contrastive)': os.path.join(MODELS_DIR, 'best_A2WNet_Contrastive.weights.h5'),
}

# Classes that form the visually ambiguous icy cluster
ICY_CLASSES = {'frost', 'glaze', 'rime', 'snow', 'dew'}

# Color palette — consistent across all panels
PALETTE = sns.color_palette('tab20', 11)


# ── Dataset ────────────────────────────────────────────────────────────────────
def build_test_gen(dataset_dir):
    df = load_dataset(dataset_dir)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    num_classes = df['label'].nunique()
    _, _, test_gen = gen(raw_identity, train_df, test_df)
    return test_gen, test_df, num_classes

dataset_dir = os.path.join(PROJECT_ROOT, 'dataset')
df_full     = load_dataset(dataset_dir)
train_df, test_df  = train_test_split(
    df_full, test_size=0.2, stratify=df_full['label'], random_state=42
)
NUM_CLASSES = df_full['label'].nunique()
CLASS_NAMES = sorted(df_full['label'].unique().tolist())
_, _, test_gen_base = gen(raw_identity, train_df, test_df)


# ── Feature extraction helpers ─────────────────────────────────────────────────

def extract_embeddings(model, test_gen, test_df, is_contrastive=False):
    """
    Run model over the full test set and collect embeddings + true labels.

    For A2WNet_Contrastive the model returns a dict; we take 'features'.
    For the other two we call model.get_embeddings().
    """
    test_gen.reset()
    steps = len(test_gen)

    all_feats  = []
    all_labels = []
    idx2cls    = {v: k for k, v in test_gen.class_indices.items()}

    for _ in range(steps):
        x, y = next(test_gen)
        if is_contrastive:
            out    = model(x, training=False)
            feats  = out['features'].numpy()
        else:
            feats  = model.get_embeddings(x).numpy()

        all_feats.append(feats)
        all_labels.extend([idx2cls[i] for i in np.argmax(y, axis=1)])

    feats  = np.vstack(all_feats)[:len(test_df)]
    labels = all_labels[:len(test_df)]

    # L2-normalise before t-SNE for a fair comparison across models
    feats  = normalize(feats, norm='l2')
    return feats, labels


def run_tsne(feats):
    print(f'  Running t-SNE on {feats.shape} ...', flush=True)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate='auto', init='pca')
    return tsne.fit_transform(feats)


# ── Load models & extract ──────────────────────────────────────────────────────

def load_model(name, num_classes, weights_path):
    print(f'\nLoading {name} ...', flush=True)
    if 'Hard Concat' in name:
        m = HybridVGGViT(num_classes=num_classes, freeze_base=False)
    elif 'Gated' in name:
        m = HybridGatedModel(num_classes=num_classes, freeze_base=False)
    else:
        m = A2WNet_Contrastive(num_classes=num_classes)

    m.compile(optimizer='adam', loss='categorical_crossentropy')
    m(tf.zeros([1, 224, 224, 3]))   # build

    if os.path.exists(weights_path):
        m.load_weights(weights_path)
        print(f'  Weights loaded: {weights_path}')
    else:
        print(f'  WARNING: weights not found at {weights_path}. Using random weights.')
    return m


model_data = {}
for title, wpath in WEIGHTS.items():
    is_contra = 'Contrastive' in title
    m         = load_model(title, NUM_CLASSES, wpath)
    _, _, tg  = gen(raw_identity, train_df, test_df)   # fresh generator each time
    feats, labels = extract_embeddings(m, tg, test_df, is_contrastive=is_contra)
    coords        = run_tsne(feats)
    model_data[title] = {'coords': coords, 'labels': labels}
    del m   # free GPU memory between models
    tf.keras.backend.clear_session()


# ── Figure 1: 3-panel full t-SNE ──────────────────────────────────────────────
print('\nPlotting Figure 1 — full t-SNE comparison ...')

color_map = {cls: PALETTE[i] for i, cls in enumerate(CLASS_NAMES)}

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for ax, (title, data) in zip(axes, model_data.items()):
    coords = data['coords']
    labels = data['labels']

    for cls in CLASS_NAMES:
        mask = [l == cls for l in labels]
        pts  = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=color_map[cls], label=cls,
                   s=18, alpha=0.7, edgecolors='none')

    ax.set_title(title.replace('\n', ' '), fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Single shared legend on the right
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color_map[c], markersize=9, label=c)
           for c in CLASS_NAMES]
axes[-1].legend(handles=handles, title='Weather Class',
                bbox_to_anchor=(1.02, 1), loc='upper left',
                fontsize=9, framealpha=0.9)

fig.suptitle('t-SNE Latent Space — Hard Concat  →  Gated  →  A2WNet (Contrastive)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out1_png = os.path.join(IMG_DIR, 'tsne_comparison.png')
out1_pdf = os.path.join(IMG_DIR, 'tsne_comparison.pdf')
plt.savefig(out1_png, dpi=300, bbox_inches='tight')
plt.savefig(out1_pdf, bbox_inches='tight')
plt.close()
print(f'  Saved → {out1_png} and {out1_pdf}')


# ── Figure 2: 3-panel icy-cluster zoom ────────────────────────────────────────
print('Plotting Figure 2 — icy cluster zoom ...')

ICY_COLORS = {
    'frost'  : '#1f77b4',
    'glaze'  : '#ff7f0e',
    'rime'   : '#2ca02c',
    'snow'   : '#9467bd',
    'dew'    : '#d62728',
}

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for ax, (title, data) in zip(axes, model_data.items()):
    coords = np.array(data['coords'])
    labels = data['labels']

    # Gray background: all non-icy classes
    non_icy_mask = [l not in ICY_CLASSES for l in labels]
    ax.scatter(coords[non_icy_mask, 0], coords[non_icy_mask, 1],
               color='#cccccc', s=10, alpha=0.3, edgecolors='none',
               label='other classes')

    # Highlighted icy classes
    for cls, color in ICY_COLORS.items():
        mask = [l == cls for l in labels]
        pts  = coords[mask]
        if pts.shape[0] == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=color, label=cls,
                   s=35, alpha=0.85, edgecolors='white', linewidths=0.3)

        # Centroid label
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.text(cx, cy, cls, fontsize=9, fontweight='bold', color=color,
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_title(title.replace('\n', ' '), fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, loc='lower right', framealpha=0.85)

fig.suptitle('Icy-Cluster Zoom — frost / glaze / rime / snow / dew\n'
             'Contrastive loss pulls same-class together and pushes different-class apart',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
out2_png = os.path.join(IMG_DIR, 'tsne_icy_cluster.png')
out2_pdf = os.path.join(IMG_DIR, 'tsne_icy_cluster.pdf')
plt.savefig(out2_png, dpi=300, bbox_inches='tight')
plt.savefig(out2_pdf, bbox_inches='tight')
plt.close()
print(f'  Saved → {out2_png} and {out2_pdf}')

print('\nDone. Both plots saved to figures/')
