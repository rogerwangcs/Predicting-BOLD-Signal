import random
import matplotlib.pyplot as plt


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
    outputImg = model.forward(inputImg.unsqueeze(0))[0].squeeze()

    fig = plt.figure(figsize=(6, 12), facecolor='white')
    fig.suptitle('Image #%d' % index, y=0.635, fontsize=16)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('Input')
    plt.imshow(inputImg.permute(1, 2, 0).detach().numpy())
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('Decoded')
    plt.imshow(outputImg.permute(1, 2, 0).detach().numpy())
    plt.show()
                                        