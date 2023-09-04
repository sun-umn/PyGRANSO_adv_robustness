from typing import List, Tuple
import torch.nn as nn
import torch
import numpy as np


# ==== Cleaned from Original Perceptual Adversarial Robustness Paper normalization ====
# ==== And add L1 norm version ====

# Note: The original papar's L2 version is actually Squared Distance:
# LPIPS_L2 = ((f1 - f2) ** 2).sum(dim=1)
# Instead of (((f1 - f2) ** 2).sum(dim=1))**(1/2)
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


def calc_distance(sample_1, sample_2, attack_type, lpips_model=None):
    if attack_type == "PAT":
        assert lpips_model is not None, "Need lpips model input"
        distance = lpips_model(sample_1, sample_2)
        return distance.item()

    delta_vec = (sample_1 - sample_2).reshape(-1)
    if attack_type == "Linf":
        distance = torch.linalg.vector_norm(delta_vec, ord=float("inf"))
    elif attack_type == "L1":
        distance = torch.linalg.vector_norm(delta_vec, ord=1)
    elif attack_type == "L2":
        distance = torch.linalg.vector_norm(delta_vec, ord=2)
    elif "L" in attack_type:
        norm_p = float(attack_type.split("L")[-1])
        distance = torch.sum(torch.abs(delta_vec)**norm_p) ** (1/norm_p)
    else:
        raise RuntimeError("Error in calculating norm")
    
    distance = distance.item()
    return distance