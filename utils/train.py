

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    avg_loss=0.0
    total=0
    
    for imgs, labels in dataloader:
        imgs, labels=imgs.to(device), labels.to(device)
        outputs=model(imgs)
        
        loss=criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss+=loss.item()*imgs.size(0)
        total+=labels.size(0)
    
    avg_loss/=total
    return avg_loss