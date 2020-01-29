import torch.nn as nn


class SimpleConvAE(nn.Module):
    def __init__(self):
        super().__init__()

        # in: b, 3, 64, 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2, return_indices=True),
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded, indices = self.encoder(x)
        out = self.unpool(encoded, indices)
        out = self.decoder(out)
        return (out, encoded)
