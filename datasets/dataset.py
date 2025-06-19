from torch.utils.data import Dataset

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset=subset
        self.transform=transform
    
    def __getitem__(self, idx):
        x, y=self.subset[idx]
        x=self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)