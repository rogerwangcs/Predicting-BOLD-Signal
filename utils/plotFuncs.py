import random
import matplotlib.pyplot as plt


def peekImageFolderDS(dataset):
    cols = 4
    rows = 2
    fig = plt.figure(figsize=(cols*3, rows*3))
    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)
        randIdx = random.randint(0, dataset.__len__())
        image, labelIdx = dataset.__getitem__(randIdx)
        plt.text(1, 5, dataset.classes[labelIdx],
                 bbox=dict(facecolor='pink', alpha=1))
        plt.imshow(image.permute(1, 2, 0))
    plt.show()
                                        