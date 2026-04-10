import torch
from torchvision import datasets, transforms
import os

def get_datasets_kfold(data_dir):
    """
    Ritorna DUE versioni del dataset: una aumentata (per il train) 
    e una pulita (per il validation test).
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

    # 1. TRASFORMAZIONI TRAINING (AUGMENTATION)
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Ruota un po' per robustezza
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. TRASFORMAZIONI VALIDATION (PULITE)
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carichiamo la cartella due volte con le due trasformazioni diverse
    dataset_train_base = datasets.ImageFolder(root=data_dir_final, transform=train_transform)
    dataset_val_base   = datasets.ImageFolder(root=data_dir_final, transform=val_transform)
    
    # FILTRA solo Tomato 
    tomato_mask = [name.startswith('Tomato') for name in dataset_train_base.classes]
    tomato_indices = [i for i, mask in enumerate(tomato_mask) if mask]
    
    valid_indices = [idx for idx, label in enumerate(dataset_train_base.targets) if label in tomato_indices]
    old_to_new_label = {old: new for new, old in enumerate(tomato_indices)}
    new_targets = [old_to_new_label[dataset_train_base.targets[i]] for i in valid_indices]
    
    classes = [dataset_train_base.classes[i] for i in tomato_indices]

    # Classe Wrapper customizzata
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
    
    # Creiamo i due dataset finali
    train_dataset = TomatoDataset(torch.utils.data.Subset(dataset_train_base, valid_indices), new_targets, classes)
    val_dataset   = TomatoDataset(torch.utils.data.Subset(dataset_val_base, valid_indices), new_targets, classes)
    
    print(f"✅ Dataset caricato per K-Fold: {len(train_dataset)} immagini, {len(classes)} classi")
    return train_dataset, val_dataset