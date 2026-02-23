"""Reads text manifest: each line is '<image_path> <label>'."""
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        with open(txt_path, 'r') as f:
            self.imgs = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.imgs.append((parts[0], int(parts[1])))
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_labels(self):
        """Returns all labels — needed for WeightedRandomSampler."""
        return [label for _, label in self.imgs]
