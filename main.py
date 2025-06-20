from models.vit_model import VisionTransformer
from utils.train import train
from utils.test import test
from utils.evaluate import evaluate
from utils.visualize import visualize_prediction
from utils.transform import get_train_tf, get_test_tf, get_basic_tf
from utils.data_utils import get_mean_std
from datasets.transform_subset import TransformSubset

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
import timm



basic_tf=get_basic_tf()

rawset=torchvision.datasets.Food101(root='/dataset/food101',split='train',transform=None, download=True)
basicset=TransformSubset(rawset, basic_tf)
basicloader=DataLoader(basicset, 32, False)
mean, std=get_mean_std(basicloader)

train_tf=get_train_tf(mean, std)
test_tf=get_test_tf(mean, std)

trainset=torchvision.datasets.Food101(root='/dataset/food101',split='train',transform=train_tf, download=True)
testset=torchvision.datasets.Food101(root='/dataset/food101',split='test',transform=test_tf, download=True)

trainloader=DataLoader(trainset, 32, True, num_workers=4, pin_memory=True)
testloader=DataLoader(testset, 16, False, num_workers=2, pin_memory=True)

"""
# Split train and validation sets (9:1)

generator=torch.Generator().manual_seed(42)

train_indices, val_indices=random_split(range(len(rawset)), [0.9, 0.1], generator=generator)

trainset=TransformSubset(Subset(rawset, train_indices), train_tf)
valset=TransformSubset(Subset(rawset, val_indices), test_tf)
testset=torchvision.datasets.Food101(root='/dataset', split='test', transform=test_tf, download=True)

trainloader=DataLoader(trainset, 32, True, num_workers=4, pin_memory=True)
valloader=DataLoader(valset, 16, False, num_workers=2, pin_memory=True)
testloader=DataLoader(testset, 16, False, num_workers=2, pin_memory=True)
"""

if __name__=="__main__":
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model=VisionTransformer().to(device)
    model=timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=101)
    model=model.to(device)
    criterion=nn.CrossEntropyLoss().to(device)
    optimizer=optim.AdamW(model.parameters(), lr=5e-5,weight_decay=1e-4)

    
    for epoch in range(40):
        train_loss=train(model, trainloader, criterion, optimizer, device)  
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.2f}')  

    test(model, testloader, device)
    visualize_prediction(model, testloader, device, mean, std)
    
    
    """
    # Split train and validation sets (9:1)
        
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=101)
    model=model.to(device)
    criterion=nn.CrossEntropyLoss().to(device)
    optimizer=optim.AdamW(model.parameters(), lr=5e-5,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3, factor=0.5)
    
    best_loss = float('inf')
    patience = 12
    early_stop_counter = 0
    save_path = "vit_best.pth"
        
    for epoch in range(40):
        train_loss=train(model, trainloader, criterion, optimizer, device)
        val_loss, acc=evaluate(model, valloader, criterion, device)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Accuracy: {acc:.2f}%')
        scheduler.step(val_loss)
        
        if val_loss<best_loss:
            best_loss=val_loss
            early_stop_counter=0
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            print(f"Best model saved!: {epoch+1}")
        else:
            early_stop_counter+=1
            print(f"Early stop counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stop: {epoch+1}")
            break
    """