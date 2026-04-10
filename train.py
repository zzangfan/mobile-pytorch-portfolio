import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CatDogDataset
from model import CatDogModel

def main():
    print("CatDog classification model training start")
    
    my_transforms = transforms.Compose([
        transforms.Resize((224,224),antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    ])
    
    train_dataset = CatDogDataset(data_dir = "./data/training_set",transform=my_transforms)
    print(f"train data numbers{len(train_dataset)}")
    
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device{device}")
    
    model = CatDogModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr =1e-3)
    
    print("Training AI starts")
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images,labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss/len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}]Complete - Average Loss:{epoch_loss:.4f}")
    
if __name__ == "__main__":
    main()