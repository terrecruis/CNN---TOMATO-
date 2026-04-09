# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

def get_dataset(data_dir):
    print(f"[🔍] Esplorando struttura dataset: {data_dir}")
    
    # 🔍 TROVA CARTELLA CON CLASSI TOMATO (ricorsivo)
    def find_tomato_folder(root):
        for dirpath, dirnames, _ in os.walk(root):
            tomato_dirs = [d for d in dirnames if d.startswith('Tomato')]
            if len(tomato_dirs) >= 8:  # Almeno 8 classi Tomato
                print(f"[✅] Dataset trovato: {dirpath}")
                print(f"   Tomato dirs: {tomato_dirs[:5]}...")
                return dirpath
        return None
    
    data_dir_final = find_tomato_folder(data_dir)
    if data_dir_final is None:
        print("❌ NESSUNA cartella Tomato trovata!")
        print("Struttura root:", os.listdir(data_dir))
        return None
    
    # Carica dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir_final, transform=transform)
    
    # ✅ FILTRA solo Tomato (auto-detect)
    tomato_mask = [name.startswith('Tomato') for name in dataset.classes]
    tomato_indices = [i for i, mask in enumerate(tomato_mask) if mask]
    
    print(f"✅ ImageFolder: {len(dataset)} img, {len(dataset.classes)} classi totali")
    print(f"✅ Tomato classes: {len(tomato_indices)}")
    
    # Subset + remap
    valid_indices = [idx for idx, label in enumerate(dataset.targets) if label in tomato_indices]
    old_to_new_label = {old: new for new, old in enumerate(tomato_indices)}
    new_targets = [old_to_new_label[dataset.targets[i]] for i in valid_indices]
    
    print(f"✅ Dataset finale: {len(valid_indices)} immagini Tomato")
    
    class TomatoDataset(torch.utils.data.Dataset):
        def __init__(self, subset_dataset, targets, classes):
            self.subset = subset_dataset
            self.targets = targets
            self.classes = classes
        
        def __len__(self):
            return len(self.targets)
        
        def __getitem__(self, idx):
            img, _ = self.subset[idx]
            return img, self.targets[idx]
    
    final_dataset = TomatoDataset(
        torch.utils.data.Subset(dataset, valid_indices), 
        new_targets, 
        [dataset.classes[i] for i in tomato_indices]
    )
    
    return final_dataset

def get_train_test_split(dataset, train_ratio=0.8):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_size, test_size], generator=generator)