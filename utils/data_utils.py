import torch

def get_mean_std(dataloader):
    mean=torch.zeros(3, dtype=torch.float32)
    std=torch.zeros(3, dtype=torch.float32)
    total=0
    
    for imgs, _ in dataloader:
        batch=imgs.size(0)
        imgs=imgs.view(batch, 3, -1)
        mean+=imgs.mean(dim=(0,2))*batch
        std+=imgs.std(dim=(0,2))*batch
        total+=batch
    mean/=total
    std/=total
    return mean, std