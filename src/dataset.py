import os

from PIL import Image
from torch.utils.data import Dataset


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            self.img_dir = os.path.join(root, "train")
        else:
            self.img_dir = os.path.join(root, "test")

        self.classes = os.listdir(self.img_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.samples = []
        for name in self.classes:
            for file in os.listdir(os.path.join(self.img_dir, name)):
                self.samples.append((os.path.join(self.img_dir, name, file), self.class_to_idx[name]))

        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)