import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
from scipy import stats
import numpy as np

from Datasets import SingleDirDataset

image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(350),
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
])


def convolveFeatures(features, sharpness):
    def hrf(t): return t ** 8.6 * np.exp(-t / 0.547)
    hrf_times = np.arange(0, 20, sharpness)
    hrf_signal = hrf(hrf_times)

    nDim = features.shape[1]  # number of features
    total_time_1 = features[:, 0].shape[0]
    total_time_2 = np.convolve(features[:, 0], hrf_signal).shape[0]

    features_conv = np.zeros([total_time_2, nDim])

    for i in range(nDim):
        features_conv[:, i] = np.convolve(features[:, i], hrf_signal)

    # Throw away the extra timepoints at the end
    features_conv = features_conv[1:total_time_1+1, :]
    return stats.zscore(features_conv, axis=0)


def encodeFeature(image, model):
    """Extract encoded values for a single image

    Arguments:
        image {Tensor} -- image dataset to sample
        model {Moduke} -- [description]

    Returns:
        Tensor -- flattened encoded image
    """
    features = model(image.unsqueeze(0).cuda())[1].cpu()
    return features.view(1, -1)  # flatten


def generateFeatures(dataset, model, featureLen):
    features = torch.empty(dataset.__len__(), featureLen)
    print('Feature Shape: %s' % str(features.shape))

    for i in tqdm_notebook(range(0, dataset.__len__())):
        inputImg = dataset.__getitem__(i)[0]
        features[i] = encodeFeature(inputImg, model)

    return features


def generateAllFeatures(model, snapshotsPaths, outputPath):

    model = model.cuda()

    for runIdx, snapshotsPath in enumerate(snapshotsPaths):

        if len(os.listdir(snapshotsPath)) < 10:
            print('Run {} has no frame snapshots. Skipping...'.format(runIdx))
            continue

        print('Encoding run {} frames...'.format(runIdx))
        dataset = SingleDirDataset(snapshotsPath, image_transforms)
        numFeatures = model(dataset.__getitem__(0)[0].unsqueeze(0).cuda())[
            1].view(1, -1).shape[1]
        trainloader = DataLoader(
            dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

        features = generateFeatures(dataset, model, numFeatures)
        torch.save(features, os.path.join(
            outputPath, 'features-{}.pt'.format(runIdx)))