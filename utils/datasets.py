import os
from torch.utils.data import Dataset
from PIL import Image


class SingleDirDataset(Dataset):
    """Forrest Dataset"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = [0]

        self.dataTable = []
        for i, image in enumerate(os.listdir(self.root)):
            imgPath = os.path.join(self.root, image)
            self.dataTable.append([imgPath, 0])

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, idx):
        sample = self.dataTable[idx]
        image = Image.open(sample[0])
        image = self.transform(image)
        return [image, sample[1]]
