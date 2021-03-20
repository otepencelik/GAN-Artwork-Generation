import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from parameters import *

# 0 : Expressionism
# 1 : Impressionism
# 2 : Realism
# 3 : Surrealism
# 4 : Byzantine
# 5 : Post-impressionism
# 6 : Symbolism
# 7 : Northern renaissance
# 8 : Suprematism
# 9 : Cubism
# 10 : Baroque
# 11 : Romanticism
# 12 : Primitism
# 13 : Mannerism
# 14 : Proto renaissance
# 15 : Early renaissance
# 16 : High renaissance
# 17 : Neoplasticism
# 18 : Pop art

labels = {"Albrecht_DuтХа├кrer" : 7,
          "Albrecht_Du╠Иrer" : 7,
          "Amedeo_Modigliani" : 0,
          "Vasiliy_Kandinskiy" : 0,
          "Diego_Rivera" : 2,
          "Claude_Monet" : 1,
          "Rene_Magritte" : 3,
          "Salvador_Dali" : 3,
          "Edouard_Manet" : 2,
          "Andrei_Rublev" : 4,
          "Vincent_van_Gogh" : 5,
          "Gustav_Klimt" : 6,
          "Hieronymus_Bosch" : 7,
          "Kazimir_Malevich" : 8,
          "Mikhail_Vrubel" : 6,          
          "Pablo_Picasso" : 9,
          "Peter_Paul_Rubens" : 10,
          "Pierre-Auguste_Renoir" : 1,
          "Francisco_Goya" : 11,
          "Frida_Kahlo" : 12,
          "El_Greco" : 13,
          "Alfred_Sisley" : 1,
          "Pieter_Bruegel" : 7,
          "Marc_Chagall" : 12,
          "Giotto_di_Bondone" : 14,
          "Sandro_Botticelli" : 15,
          "Caravaggio" : 10,
          "Leonardo_da_Vinci" : 16,
          "Diego_Velazquez" : 10,
          "Henri_Matisse" : 1,
          "Jan_van_Eyck" : 7,
          "Edgar_Degas" : 1,
          "Rembrandt" : 10,
          "Titian" : 16,
          "Henri_de_Toulouse-Lautrec" : 5,
          "Gustave_Courbet" : 2,
          "Camille_Pissarro" : 1,
          "William_Turner" : 11,
          "Edvard_Munch" : 6,
          "Paul_Cezanne" : 5,
          "Eugene_Delacroix" : 11,
          "Henri_Rousseau" : 12,
          "Georges_Seurat" : 5,
          "Paul_Klee" : 0,
          "Piet_Mondrian" : 17,
          "Joan_Miro" : 3,
          "Andy_Warhol" : 18,
          "Paul_Gauguin" : 6,
          "Raphael" : 16,
          "Michelangelo" : 16,
          "Jackson_Pollock" : 0}

class StyleDataset(Dataset):

    def __init__(self, n_class=n_class, transforms_=None):
        self.data    = torchvision.datasets.ImageFolder(root=dataroot)
        self.n_class = n_class
        
        self.transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
#                                       transforms.CenterCrop(image_size),
#                                       transforms.RandomCrop(32, padding=2),
#                                       transforms.RandomHorizontalFlip(),                                    
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = self.data.samples[idx][0]
        img = Image.open(img_name).convert('RGB')
        
        artist_name = img_name.split('/')[5]
        artist_name = artist_name.rstrip(".jpg")
        artist_name = artist_name.rstrip("_0123456789")        
        
        label = labels[artist_name]
        
        #img = np.asarray(img) / 255. # scaling [0-255] values to [0-1]
        #label = np.asarray(label)
        
        img = self.transforms(img).float()

        # create one-hot encoding
        target = torch.zeros(self.n_class, 1, dtype=torch.long)
        target[label] = 1
        
        return img, target, label
    
    
    
    
    
    
    
    
    
    
    