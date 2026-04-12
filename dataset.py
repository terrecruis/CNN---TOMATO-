import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

def get_dataset(data_dir):
    """
    Ritorna il dataset originale (per il vecchio Main - Fase 1).
    """
    print(f"[🔍] Esplorando struttura dataset: {data_dir}")
    
    def find_tomato_folder(root):
        for dirpath, dirnames, _ in os.walk(root):
            tomato_dirs = [d for d in dirnames if d.startswith('Tomato')]
            if len(tomato_dirs) >= 8:
                return dirpath
        return None
    
    data_dir_final = find_tomato_folder(data_dir)
    if data_dir_final is None:
        raise FileNotFoundError("❌ NESSUNA cartella Tomato trovata!")

    # Trasformazioni base
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir_final, transform=transform)
    
    # Filtra solo Tomato
    tomato_mask = [name.startswith('Tomato') for name in full_dataset.classes]
    tomato_indices = [i for i, mask in enumerate(tomato_mask) if mask]
    
    valid_indices = [idx for idx, label in enumerate(full_dataset.targets) if label in tomato_indices]
    old_to_new_label = {old: new for new, old in enumerate(tomato_indices)}
    new_targets = [old_to_new_label[full_dataset.targets[i]] for i in valid_indices]
    
    classes = [full_dataset.classes[i] for i in tomato_indices]

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
    
    filtered_dataset = TomatoDataset(torch.utils.data.Subset(full_dataset, valid_indices), new_targets, classes)
    return filtered_dataset

def get_train_test_split(dataset, train_ratio=0.8):
    """
    Divide il dataset in Train e Test (Holdout - Fase 1).
    """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], 
                                         generator=torch.Generator().manual_seed(42))
    return train_data, test_data

def get_datasets_kfold(data_dir):
    """
    Ritorna DUE versioni del dataset: aumentata e pulita (Per K-Fold - Fase 2).
    """
    print(f"[🔍] Esplorando struttura dataset (K-Fold): {data_dir}")
    
    def find_tomato_folder(root):
        for dirpath, dirnames, _ in os.walk(root):
            tomato_dirs = [d for d in dirnames if d.startswith('Tomato')]
            if len(tomato_dirs) >= 8:
                return dirpath
        return None
    
    data_dir_final = find_tomato_folder(data_dir)
    if data_dir_final is None:
        raise FileNotFoundError("❌ NESSUNA cartella Tomato trovata!")

    # 1. TRASFORMAZIONI TRAINING (AUGMENTATION)
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. TRASFORMAZIONI VALIDATION (PULITE)
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train_base = datasets.ImageFolder(root=data_dir_final, transform=train_transform)
    dataset_val_base   = datasets.ImageFolder(root=data_dir_final, transform=val_transform)
    
    # Filtra solo Tomato 
    tomato_mask = [name.startswith('Tomato') for name in dataset_train_base.classes]
    tomato_indices = [i for i, mask in enumerate(tomato_mask) if mask]
    
    valid_indices = [idx for idx, label in enumerate(dataset_train_base.targets) if label in tomato_indices]
    old_to_new_label = {old: new for new, old in enumerate(tomato_indices)}
    new_targets = [old_to_new_label[dataset_train_base.targets[i]] for i in valid_indices]
    
    classes = [dataset_train_base.classes[i] for i in tomato_indices]

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
    
    train_dataset = TomatoDataset(torch.utils.data.Subset(dataset_train_base, valid_indices), new_targets, classes)
    val_dataset   = TomatoDataset(torch.utils.data.Subset(dataset_val_base, valid_indices), new_targets, classes)
    
    print(f"✅ Dataset caricato per K-Fold: {len(train_dataset)} immagini, {len(classes)} classi")
    return train_dataset, val_dataset