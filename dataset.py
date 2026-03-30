import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

def get_dataset(data_dir):
    """
    Carica il dataset PlantVillage applicando le trasformazioni necessarie.
    Se kagglehub estrae una cartella padre (es. 'plantvillage'), aggiungiamo la logica 
    per puntare automaticamente alla cartella corretta con le 10 classi.
    """
    # Controllo se le cartelle sono dentro una sottocartella (es. 'plantvillage')
    if "plantvillage" in os.listdir(data_dir):
        data_dir = os.path.join(data_dir, "plantvillage")
        
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset

def get_train_test_split(dataset, train_ratio=0.8):
    """
    Divide il dataset dinamicamente usando una proporzione (es. 80% train, 20% test).
    Questo garantisce di usare tutte le ~14.500 immagini, soddisfacendo 
    il vincolo della traccia (>10.000 train e >2.500 test).
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
        
    # Usiamo un generatore con seed fisso in modo che la divisione sia sempre 
    # la stessa ad ogni avvio del programma (fondamentale per i test)
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=generator
    )
    
    return train_dataset, test_dataset