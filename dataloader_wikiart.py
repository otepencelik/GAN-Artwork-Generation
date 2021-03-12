import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from parameters import *

def get_dataset():
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
#                                     transforms.CenterCrop(image_size),
#                                    transforms.RandomCrop(32, padding=2),
#                                    transforms.RandomHorizontalFlip(),                                    
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(root=dataroot,transform=transform)

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=workers, pin_memory=True)