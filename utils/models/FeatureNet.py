import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):

    def __init__(self, featuresLen, outputLen):
        super(FeatureNet, self).__init__()
        self.fc1 = nn.Linear(featuresLen, outputLen)

    def forward(self, x):
        x = self.fc1(x)
        return x
