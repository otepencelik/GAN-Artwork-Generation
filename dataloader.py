import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def get_dataset(image_size, batch_size):
    transform = transforms.Compose([transforms.Resize(image_size),
#                                    transforms.RandomCrop(32, padding=2),
#                                    transforms.RandomHorizontalFlip(),                                    
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(root="../Datasets/BestArtworks/resized",transform=transform)

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    