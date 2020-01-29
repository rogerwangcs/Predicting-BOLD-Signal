import torch
from tqdm import tqdm_notebook


def encodeFeature(image, model):
    """Extract encoded values for a single image

    Arguments:
        image {Tensor} -- image dataset to sample
        model {Moduke} -- [description]

    Returns:
        Tensor -- flattened encoded image
    """
    netOutput = model.forward(image.unsqueeze(0))
    features = netOutput[1]
    return features.view(1, -1)  # flatten


def generateFeatures(dataset, model, featureLen):
    features = torch.empty(dataset.__len__(), featureLen)
    print('Feature Shape: %s' % str(features.shape))

    for i in tqdm_notebook(range(0, dataset.__len__())):
        inputImg = dataset.__getitem__(i)[0]
        features[i] = encodeFeature(inputImg, model)

    return features
