from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
from parameters import *

n_class = 27

class WikiartDataset(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms_=None):
        self.data      = pd.read_csv(csv_file, header=None)
        self.n_class   = n_class
        self.mode = csv_file
        
        # Add any transformations here
        #self.train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=1),])
        
        # The following transformation normalizes each channel using the mean and std provided
        self.transforms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = self.data.iloc[idx, 0]
        # Might need something like this here: img_name = 'Wikiart/' + img_name
        img = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        
        #img = np.asarray(img) / 255. # scaling [0-255] values to [0-1]
        
        img = self.transforms(img).float() # Normalization

        # create one-hot encoding
        target = torch.zeros(self.n_class, 1, dtype=torch.long)
        target[label] = 1
        
        return img, target, label
    
    
    
    
    
    
    
    
    
    