import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from datasets import SingleDirDataset
from fitFuncs import fitAE
from plotFuncs import peekImageFolderDS, sampleAE
from utils import datasets, plotFuncs, SimpleConvAE, extractRunFrames, generateFeatures
from torch.utils.data import DataLoader
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

"""# Train Autoencoder"""

HundredObjectsDSDir = "datasets/101_ObjectCategories"
image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
])
traindataset = torchvision.datasets.ImageFolder(
    root=HundredObjectsDSDir,
    transform=image_transforms
)
trainloader = DataLoader(traindataset, batch_size=16,
                         shuffle=True, num_workers=8, pin_memory=True)

# peekImageFolderDS(traindataset)


aeNet = SimpleConvAE().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(aeNet.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=0.0001)
# fitAE(aeNet, criterion, optimizer, trainloader, total_epochs=10)

# Save to 'model.pth'
torch.save({
    'model_state_dict': aeNet.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, './models/SimpleConvAE.pth')
