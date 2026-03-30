# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

def get_dataset(data_dir):
    """
    Carica il dataset PlantVillage saltando eventuali cartelle duplicate.
    """
    # Se il path attuale contiene una sottocartella chiamata 'plantvillage'
    # che a sua volta contiene le classi, entriamo direttamente lì.
    potential_path = os.path.join(data_dir, "plantvillage")
    
    # Controlliamo se dentro 'plantvillage' ci sono le cartelle Tomato
    if os.path.exists(potential_path) and any(d.startswith("Tomato") for d in os.listdir(potential_path)):
        data_dir = potential_path
        print(f"[Info] Puntatore dataset aggiornato a: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ora ImageFolder leggerà SOLO le cartelle dentro 'plantvillage'
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset

def get_train_test_split(dataset, train_ratio=0.8):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_size, test_size], generator=generator)