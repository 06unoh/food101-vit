import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_prediction(model, dataloader, device, mean, std):
    model.eval()
    mean=np.array(mean)
    std=np.array(std)
    
    with torch.no_grad():
        dataiter=iter(dataloader)
        imgs, labels=next(dataiter)
        imgs, labels=imgs.to(device), labels.to(device)
        outputs=model(imgs)
        _, preds=torch.max(outputs, dim=1)
    
    figs, axes=plt.subplots(3, 3, figsize=(8, 8))
    axes=axes.flatten()
    
    for i in range(9):
        img=imgs[i].cpu().permute(1, 2, 0).numpy()
        img=img*std+mean
        img=np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'pred: {preds[i].item()}, label: {labels[i].item()}')
    plt.tight_layout()
    plt.show()
        
        
    