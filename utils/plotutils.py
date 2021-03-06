import random
import numpy as np
import matplotlib.pyplot as plt
from utils import paths


def peekImageFolderDS(dataset):
    cols = 4
    rows = 2
    fig = plt.figure(figsize=(cols*3, rows*3), facecolor='white')
    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)
        randIdx = random.randint(0, dataset.__len__())
        image, labelIdx = dataset.__getitem__(randIdx)
        plt.text(1, 5, dataset.classes[labelIdx],
                 bbox=dict(facecolor='pink', alpha=1))
        plt.imshow(image.permute(1, 2, 0))
    plt.show()
    return fig


def sampleAE(dataset, model, index=None):
    """ Plots the input image and output image of an autoencoder side by side

    Keyword arguments:
    dataset -- torch dataset object to select images from
    model == torch nn model to feed image
    index -- optionally specifies dataset item index to display
    """
    if not index:
        index = random.randint(0, dataset.__len__())

    inputImg = dataset.__getitem__(index)[0]
    outputImg = model.forward(inputImg.cuda().unsqueeze(0))[0].squeeze().cpu()

    fig = plt.figure(figsize=(6, 12), facecolor='white')
    fig.suptitle('Image #%d' % index, y=0.635, fontsize=16)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('Input')
    plt.imshow(inputImg.permute(1, 2, 0).detach().numpy())
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('Decoded')
    plt.imshow(outputImg.permute(1, 2, 0).detach().numpy())
    plt.show()
    return fig


def plotCompareFeatures(feature, feature_conv):
    fig1, (ax1, ax2) = plt.subplots(2, 1)
    fig1.set_facecolor('white')
    ax1.plot(feature)
    ax2.plot(feature_conv)
    fig1.show()
    return fig1


def showAnat(fdata, coords):
    """ Function to display row of image slices """
    slice_0 = fdata[coords[0], :, :, coords[3]]
    slice_1 = fdata[:, coords[1], :, coords[3]]
    slice_2 = fdata[:, :, coords[2], coords[3]]
    slices = [slice_0, slice_1, slice_2]

    fig, axes = plt.subplots(1, len(slices))
    fig.set_facecolor('white')
    labels = ['x', 'y', 'z']
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].set_xlabel(labels[i])
    fig.align_labels()
    fig.show()
    return fig

# showAnat(anat_data, (45, 27, 17, 0))
# repeated_mask_data=np.repeat(mask_data[:, :, :, np.newaxis], 451, axis=3) # duplicate for 451 time frames
# showAnat(repeated_mask_data, (45, 27, 17, 1))
# anat_masked = anat_data * repeated_mask_data
# showAnat(anat_masked, (45, 27, 17, 1))

def get_results_col_order(cv_results_path):
    cv_results = np.load(cv_results_path)
    train_test_results = np.empty((8,2))
    for test_run_idx in range(8):
        total_corr = np.sum(cv_results[0:test_run_idx, test_run_idx]) \
                   + np.sum(cv_results[test_run_idx+1:, test_run_idx])
        train_test_results[test_run_idx, 0] = total_corr/7
        train_test_results[test_run_idx, 1] = cv_results[test_run_idx, test_run_idx]
    return train_test_results

def get_results_row_order(cv_results_path):
    cv_results = np.load(cv_results_path)
    train_test_results = np.empty((8,2))
    for test_run_idx in range(8):
        total_corr = np.sum(cv_results[test_run_idx, 0:test_run_idx]) \
                   + np.sum(cv_results[test_run_idx, test_run_idx+1:])
        train_test_results[test_run_idx, 0] = total_corr/7
        train_test_results[test_run_idx, 1] = cv_results[test_run_idx, test_run_idx]
    return train_test_results

def plot_all_results(subjects, feature_models, plot_shape, col_idx, results_save_name):
    rows, cols = plot_shape
    print(rows)
    fig = plt.figure(figsize=(rows*8, cols*3), facecolor='white')
    for idx, sub in enumerate(subjects):
        fig.add_subplot(rows, cols, idx+1)
        for feature_model in feature_models:
            cv_results_path = ''
            if feature_model == 'LR':
                cv_results_path = paths.results_path + "/sub-{:02d}/{}-{}.npy".format(sub, feature_model, 'march29')
            else:
                cv_results_path = paths.results_path + "/sub-{:02d}/{}-{}.npy".format(sub, feature_model, 'march29')

            cv_results = get_results_row_order(cv_results_path)

            plt.plot(cv_results[:,col_idx])

        plt.title("Subject {:02d}".format(sub))
        plt.legend(feature_models)
        plt.xlabel("Run #")
        plt.ylabel("Correlation")
    plt.show()
