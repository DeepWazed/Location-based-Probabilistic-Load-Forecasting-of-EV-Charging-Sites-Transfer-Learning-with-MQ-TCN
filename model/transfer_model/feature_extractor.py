import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_index):
        super(FeatureExtractor, self).__init__()

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(model.children())[0][:layer_index])

    def forward(self, x):
        x = self.features(x)
        x = x.transpose(1, 2)
        return x


def extract_features(x, feature_extractor):
    features = []
    with torch.no_grad():
        outputs = feature_extractor(x.permute(0, 2, 1))
        features.append(outputs)
    features = torch.cat(features, dim=0)
    return features