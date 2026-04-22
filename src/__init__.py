"""
A2WNet: Adaptive Attention Weather Network
Model architectures for weather image recognition.
"""

from .vgg16 import VGG16Classifier
from .alexnet import AlexNetClassifier
from .resnet import create_resnet_model
from .mobilenet import create_mobilenet_model
from .vit import ViTClassifier
from .hybrid_vgg_vit import HybridVGGViT
from .hybrid_gated import HybridGatedModel
from .hybrid_contrastive import A2WNet_Contrastive, SupervisedContrastiveLoss

