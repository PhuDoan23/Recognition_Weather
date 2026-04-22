"""
A2WNet: Adaptive Attention Weather Network
Model architectures for weather image recognition.
"""

from .baselines.vgg16 import VGG16Classifier
from .baselines.alexnet import AlexNetClassifier
from .baselines.resnet import create_resnet_model
from .baselines.mobilenet import create_mobilenet_model
from .baselines.vit import ViTClassifier
from .contributions.hybrid_vgg_vit import HybridVGGViT
from .contributions.hybrid_gated import HybridGatedModel
from .contributions.hybrid_contrastive import A2WNet_Contrastive, SupervisedContrastiveLoss

