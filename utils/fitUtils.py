import os
from tqdm import tqdm_notebook
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import BatchSampler, RandomSampler, ConcatDataset, DataLoader

from models import FeatureNet
from evalUtils import featuresCorr
from plotUtils import  sampleAE
from evalUtils import correlateDataset
from loadUtils import loadModel
from featureUtils import generateAllFeatures
import Paths


def fitAE(model, criterion, optimizer, dataloader, epochs=10):
    model.train()
    loss_memory = []
    pbar = tqdm_notebook(range(epochs), leave=True)

    numBatches = len(dataloader)
    # set running loss print frequency
    lossCutoff = min(numBatches, (numBatches*epochs)/100)

    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            (inputBatch, labelBatch) = batch
            inputBatch = inputBatch.cuda()
            labelBatch = labelBatch.cuda()

            # forward
            optimizer.zero_grad()
            outputBatch = model(inputBatch.float())[0]
            loss = criterion(outputBatch, inputBatch)

            # backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % lossCutoff == 0:
                pbar.set_description("Loss: %.4f" % (running_loss))
                pbar.refresh()
                running_loss = 0.0


def fitFeatureModel(model, criterion, optimizer, dataloader, epochs=10, debug=False):
    model.train()
    loss_memory = []

    pbar = range(epochs)
    if debug:
        pbar = tqdm_notebook(range(epochs), leave=True)

    numBatches = len(dataloader)
    # set running loss print frequency
    lossCutoff = min(numBatches, (numBatches*epochs)/100)

    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            (inputBatch, labelBatch) = batch
            inputBatch = inputBatch.cuda()
            labelBatch = labelBatch.cuda()

            # forward
            optimizer.zero_grad()
            outputBatch = model(inputBatch.float())
            loss = criterion(outputBatch, labelBatch.float())

            # backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % lossCutoff == 0 and debug:
                pbar.set_description("Loss: %.4f" % (running_loss))
                pbar.refresh()
                running_loss = 0.0
                

def featureNetCrossVal(datasets, outputPath, epochs, debug=False):
    eval_scores = np.empty((8, 2))
    for i in range(8):
        # create indices for train and test split
        train_runs_indices = [j for j in range(8)]
        test_run = train_runs_indices.pop(i)
        train_dataset_list = [datasets[i] for i in train_runs_indices]

        # create combined training set
        combined_train_dataset = ConcatDataset(train_dataset_list)
        train_sampler = RandomSampler(combined_train_dataset)
        train_loader = DataLoader(combined_train_dataset, sampler=train_sampler, batch_size=64, num_workers=0, pin_memory=True)

        # create model, optimizer, error function
        num_features = combined_train_dataset.__getitem__(0)[0].shape[0]
        num_ffa_voxels = combined_train_dataset.__getitem__(0)[1].shape[0]
        model = FeatureNet(num_features, num_ffa_voxels).cuda()
        model.train()
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

        # fit model
        fitFeatureModel(model, criterion, optimizer, train_loader, epochs=epochs, debug=debug)
        
        # Evaluate model
        model.eval()
        # in sample error
        eval_scores[i, 0] = featuresCorr(combined_train_dataset, model)
        # out of sample error
        test_dataset = datasets[i]
        eval_scores[i, 1] = featuresCorr(test_dataset, model)
        
        # Save data
        np.save(outputPath, eval_scores)
    return eval_scores

def train_all_autoencoders(dataloader, autoencoders, epochs=50, save_name=''):
    for ae in autoencoders:
        # instantiate and fit model
        print('Training {} model...'.format(ae))
        model = autoencoders[ae]
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        fitAE(model, criterion, optimizer, dataloader, epochs=epochs)

        # show size of encoded featuers
        encoded_features_size = model(dataloader.dataset.__getitem__(0)[0].unsqueeze(0).cuda())[1].view(1, -1).shape[1]
        print("Features Size: {}".format(encoded_features_size))

        # save sample image
        fig = sampleAE(dataloader.dataset, model)
        fig_path = os.path.join(Paths.models_path, 'autoencoders/{}-{}.png'.format(ae, save_name))
        fig.savefig(fig_path)
        
        # calculate correlation between dataset
        corr = correlateDataset(dataloader.dataset, model)
        print(corr)
        
        # save model
        model_params = {
            'model_state_dict': model.state_dict()
        }
        save_path = os.path.join(Paths.models_path, 'autoencoders/{}-{}.pth'.format(ae, save_name))
        print("{} model saved.".format(ae))
        torch.save(model_params, save_path)
        
        print("Training complete.")
        
def encode_all_forrest_mult_autoencoders(autoencoders, save_name=''):
    for ae in autoencoders:
        
        # Load model for eval
        model = autoencoders[ae]
        model_path = os.path.join(Paths.models_path, 'autoencoders/{}-{}.pth'.format(ae, save_name))
        loadModel(model_path, model)
        model.eval()
        
        # Generate features for all forrest runs for one ae
        print('Encoding forrest features for {} model...'.format(ae))
        save_path = Paths.encoded_features(ae, save_name=save_name)
        snapshot_paths = [Paths.movie_run_snapshots_path(i) for i in range(8)]
        generateAllFeatures(model, snapshot_paths, save_path, save_name=save_name)
        
        print("Feature encoding complete.")