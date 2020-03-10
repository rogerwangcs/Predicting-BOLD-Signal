import torch
import numpy as np

# predictFeature : returns 1d numpy array of response prediction in voxels
def predictFeature(dataset, model, idx):
    feature = dataset.__getitem__(idx)[0]
    label = dataset.__getitem__(idx)[1]
    pred = model.forward(feature.cuda()).cpu().detach().numpy()
    return pred, label

def predictAllFeatures(dataset, model):
    num_time_frames = len(dataset)
    num_ffa_voxels = dataset.__getitem__(0)[1].shape[0]
    
    preds_and_labels = np.zeros((num_time_frames, num_ffa_voxels, 2))
    
    for i in range(num_time_frames):
        sample_pred = predictFeature(dataset, model, i)
        preds_and_labels[i, :, 0] = sample_pred[0]
        preds_and_labels[i, :, 1] = sample_pred[1]
    
    return preds_and_labels

    return np.corrcoef(x[:, 0], x[:, 1])[0, 1]


def featuresCorr(dataset, model):
    """ Returns average correlation of output of given linear model vs true brain response """
    
    def featureAcrossTimeCorr(x):
        return np.corrcoef(x[:, 0], x[:, 1])[0, 1]

    preds_and_labels = predictAllFeatures(dataset, model)
    corrs = [featureAcrossTimeCorr(x) for x in preds_and_labels]
    return sum(corrs)/len(corrs)

def correlateDataset(dataset, model):
    """ Returns average correlation of the images in given dataset vs its autoencoded output of given model """
    
    corrs = np.empty(len(dataset))
    
    def correlateImage(img1, img2):
        img_size = img1.shape[1] # (3 [64] 64)
        x = img2 - img1
        x = x ** 2
        x = np.sum(x, axis=0)
        x = np.sqrt(x)
        return 1 - np.sum(x)/(img_size**2)

    for i in range(len(dataset)):
        inputImg = dataset.__getitem__(i)[0]
        outputImg = model.forward(inputImg.cuda().unsqueeze(0))[0].squeeze().cpu()
        inputImg = inputImg.numpy()
        outputImg = outputImg.detach().numpy()
        corrs[i] = correlateImage(inputImg, outputImg)
    return np.mean(corrs)