import os
import torch
import nibabel as nib
from scipy import stats

import Paths


def loadModel(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
def loadFeatures(model, sub, run, responseLen):
    features = torch.load(Paths.encoded_features(model, sub, run))
    features = features[:responseLen]
    features = stats.zscore(features.detach(), axis=0)
    print('Movie Run Frames: {}'.format(features.shape[0]))
    return features


def loadFMRI(sub, run):
    anat_data = nib.load(Paths.func_path(sub, run)).get_fdata()
    mask_data = nib.load(Paths.ffa_mask_path(sub)).get_fdata()
    print('FMRI Frames: {}'.format(anat_data.shape[3]))
    return anat_data, mask_data
