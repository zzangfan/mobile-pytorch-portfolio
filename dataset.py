import torch
import os
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms


class CatDogDataset(Dataset):
    
    def __init__(self,data_dir,transform = None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        class_to_label = {"cats":0,
                          "dogs":1}
        
        for class_name, label in class_to_label.items():
            folder_path = os.path.join(data_dir,class_name)
            if not os.path.exists(folder_path):
                continue
            
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith((".png",".jpg","jpeg")):
                    self.image_paths.append(os.path.join(folder_path,file_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = torchvision.io.read_image(img_path)
       
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

