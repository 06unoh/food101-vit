import torch

def test(model, dataloader, device):
    model.eval()
    total=0
    correct=0
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels=imgs.to(device), labels.to(device)
            outputs=model(imgs)
            
            _, preds=torch.max(outputs, dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
            
    acc=(correct*100)/total    
    print(f'Accuracy: {acc:.2f}%')