import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total=0
    correct=0
    avg_loss=0.0
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels=imgs.to(device), labels.to(device)
            outputs=model(imgs)
            loss=criterion(outputs, labels)
            _, preds=torch.max(outputs, dim=1)
            
            avg_loss+=loss.item()*imgs.size(0)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    
    avg_loss/=total      
    acc=(correct*100)/total    
    return avg_loss, acc