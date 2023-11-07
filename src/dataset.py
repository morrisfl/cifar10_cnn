import os

from PIL import Image
from torch.utils.data import Dataset


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.img_dir = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []

        for name in self.classes:
            for file in os.listdir(os.path.join(root, name)):
                self.samples.append((os.path.join(root, name, file), self.class_to_idx[name]))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)