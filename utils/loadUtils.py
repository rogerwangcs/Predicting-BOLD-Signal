import os
import numpy as np
import torch
import nibabel as nib
from scipy import stats
import torchvision

from Datasets import SingleDirDataset

image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(350),
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
])

import Paths


def loadModel(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
def loadFeatures(model, run, responseLen, debug=False):
    features = torch.load(Paths.encoded_features(model, run))
    features = features[:responseLen]
    features = stats.zscore(features.detach(), axis=0)
    if debug:
        print('Movie Run Frames: {}'.format(features.shape[0]))
    return features


def loadFMRI(sub, run, debug=False):
    anat_data = nib.load(Paths.func_path(sub, run)).get_fdata()
    mask_data = nib.load(Paths.ffa_mask_path(sub)).get_fdata()
    if debug:
        print('FMRI Frames: {}'.format(anat_data.shape[3]))
    return anat_data, mask_data



def loadPixelFeatures(snapshotsPath, run, responseLen, debug=False):
    
    dataset = SingleDirDataset(snapshotsPath, image_transforms)
    featuresLen = torch.flatten(dataset.__getitem__(0)[0]).shape[0]
    features = np.empty(shape=(len(dataset), featuresLen))
    for i in range(len(dataset)):
        sample = torch.flatten(dataset.__getitem__(i)[0])
        sample = sample.numpy()
        features[i] = sample
    
    features = features[:responseLen]
    features = stats.zscore(features, axis=0)
    if debug:
        print('Movie Run Frames: {}'.format(features.shape[0]))
    return features