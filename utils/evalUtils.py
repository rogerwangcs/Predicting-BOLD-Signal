import torch
import numpy as np

# def correlateFeature(dataset, model,  idx):
#     feature = dataset.__getitem__(idx)[0]
#     label = dataset.__getitem__(idx)[1]
#     pred = model.forward(torch.tensor(feature).float().cuda()).cpu()

#     pearsonCorr = torch.tensor(np.corrcoef(pred.clone().detach().numpy(), label.numpy())).float()
#     return pearsonCorr[0, 1].item()

# def evaluateFeatureDataset(dataset, model):
#     scores = []
#     for i in range(len(dataset)):
#         score = correlateFeature(dataset, model, i)
#         scores.append(score)
#     return sum(scores)/len(scores)


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

def featureAcrossTimeCorr(x):
    return np.corrcoef(x[:, 0], x[:, 1])[0, 1]


def featuresCorr(dataset, model):
    preds_and_labels = predictAllFeatures(dataset, model)
    corrs = [featureAcrossTimeCorr(x) for x in preds_and_labels]
    return sum(corrs)/len(corrs)