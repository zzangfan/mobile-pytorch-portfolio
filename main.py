# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 우리가 만든 dataset.py 파일에서 CatDogDataset 클래스(직원)를 수입해온다!
from dataset import CatDogDataset
from model import CatDogModel


def main():
    
    my_transform = transforms.Compose([
        transforms.Resize((224,224),antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    ])
    
    train_dataset = CatDogDataset("./data/training_set",transform= my_transform)
    
    test_dataset = CatDogDataset("./data/test_set",transform=my_transform)
    
    print(f"총 데이터 개수{len(test_dataset)}장")
    
    #테스트용 데이터로더
    test_dataloader= DataLoader(
        dataset = test_dataset,
        batch_size = 32,
        shuffle= True,
        num_workers= 0
    )
    
    for images, labels in test_dataloader:
        print("--- 첫 번째 데이터 도착----")
        print(f"사진 형태 (Batch, Channels, Height, Width):{images.shape}")
        print(f"정답지 목록:{labels}")
        break
if __name__ =="__main__":
    main()
    
    