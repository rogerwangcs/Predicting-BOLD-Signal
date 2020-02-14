import os
import torch
import nibabel as nib
from scipy import stats


def loadModel(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


studyforrest_path = './datasets/studyforrest'


def ffa_mask_path(sub): return os.path.join(studyforrest_path,
                                            'sub-0{}_ROIs/rFFA_final_mask_sub-0{}_bin.nii.gz'.format(sub, sub))


def func_path(sub, run): return os.path.join(studyforrest_path,
                                             'sub0{}_ses-movie/ses-movie/func_newSize/sub-0{}_ses-movie_task-movie_run-{}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz').format(sub, sub, run + 1)


def features_path(sub, run): return os.path.join(
    studyforrest_path, 'encoded_features/features-{}.pt'.format(run))


def loadFeatures(sub, run, responseLen):
    features = torch.load(features_path(sub, run))
    features = features[:responseLen]
    features = stats.zscore(features.detach(), axis=0)
    print('Movie Run Frames: {}'.format(features.shape[0]))
    return features


def loadFMRI(sub, run):
    anat_data = nib.load(func_path(sub, run)).get_fdata()
    mask_data = nib.load(ffa_mask_path(sub)).get_fdata()
    print('FMRI Frames: {}'.format(anat_data.shape[3]))
    return anat_data, mask_data
