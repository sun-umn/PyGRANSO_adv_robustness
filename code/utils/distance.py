from typing import List, Tuple
import torch.nn as nn
import torch
import numpy as np


# ==== Original Perceptual Adversarial Robustness Paper normalization ====
# ==== Subject to future changes
def normalize_flatten_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
) -> torch.Tensor:
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """
    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)


def normalize_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
):
    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
        )
    return normalized_features


def unify_features(
    features: Tuple[torch.Tensor, ...],
    eps=1e-10,
):
    normalized_features: List[torch.Tensor] = []
    for feature_layer in features:
        feature_layer = feature_layer.view(
            feature_layer.shape[0],
            feature_layer.shape[1],
            -1
            )
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=2, keepdim=True)) + eps
        factor_2 = feature_layer.shape[1]
        normalized_features.append(
            (feature_layer / (norm_factor * factor_2))
        )
    return normalized_features


class LPIPSDistance(nn.Module):
    """
        Modified LPIPS Distance class: delete redundent functionalities.
    """

    def __init__(
        self,
        model,
        activation_distance="L2",
    ):
        """
            Constructs an LPIPS distance metric. The given network should return a
            tuple of (activations, logits).

            model must have .features function to output a tuple of latent features.

        """
        super().__init__()
        self.model = model

        self.activation_distance = activation_distance
        self.eval()

    def features(self, image):
        return self.model.features(image)

    def forward(self, image1, image2):
        features1 = self.features(image1)
        features2 = self.features(image2)

        f1 = normalize_flatten_features(features1)
        f2 = normalize_flatten_features(features2)
        if self.activation_distance == 'L2':
            res = ((f1 - f2) ** 2).sum(dim=1)
            return res
        elif self.activation_distance == "L1":
            res = torch.abs(f1 - f2).sum(dim=1)
            return res
        else:
            raise ValueError(
                f'Invalid activation_distance "{self.activation_distance}"')
    
    def visualization_forward(self, image1, image2):
        """
            Return a list of feature distance --- to show scaling issue.
        """
        features1 = self.features(image1)
        features2 = self.features(image2)
        f1 = normalize_features(features1)
        f2 = normalize_features(features2)

        if self.activation_distance == 'L2':
            distance = []
            for idx in range(len(f1)):
                distance.append(((f1[idx]-f2[idx])**2).sum().item())
            return distance
        elif self.activation_distance == "L1":
            distance = []
            for idx in range(len(f1)):
                distance.append(torch.abs(f1[idx]-f2[idx]).sum().item())
            return distance
        else:
            raise ValueError(
                f'Invalid activation_distance "{self.activation_distance}"')

    

if __name__ == "__main__":
    import sys, os
    sys.path.append("/home/hengyue/MinMaxGranso/code")
    sys.path.append("E:\\MinMaxGranso\\code")
    from models.model import AlexNetFeatureModel
    os.system("clear")

    test_model = AlexNetFeatureModel()

    a1 = torch.randn(size=(4, 3, 224, 224))
    a2 = torch.rand_like(a1)

    metric = LPIPSDistance(
        model=test_model
    )

    distance = metric(a1, a2)
    print(distance.shape)

    print()
