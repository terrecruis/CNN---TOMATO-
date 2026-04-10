import torch
import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import random

from dataset import get_datasets_kfold
from model import TomatoCNN, TomatoFCNN
from train import train_stratified_kfold
# ... mantieni gli altri import

def set_seed(seed=42):
    """Fissa il seed per rendere i risultati 100% riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed impostato a {seed} per riproducibilità.")

def main():
    set_seed(42) # Chiamiamo il seed come primissima cosa
    
    print("="*55)
    print("  P14 - FCNN vs CNN su PlantVillage (Stratified K-Fold)")
    print("="*55)

    dataset_path = kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")
    
    # Recuperiamo DUE dataset: uno con augmentation, uno senza.
    train_dataset, val_dataset = get_datasets_kfold(dataset_path)

    # ... la tua configurazione per l'esperimento ...
    
    # ESEMPIO DI CHIAMATA PER LA CNN:
    # Definiamo i parametri da passare alla funzione k-fold
    cnn_kwargs = {'n_filters': 32, 'kernel_size': 3, 'num_blocks': 3, 'num_classes': 10}
    
    print("\n-> Esecuzione K-Fold su CNN Ottimale:")
    results, last_model = train_stratified_kfold(
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        model_class=TomatoCNN,
        model_kwargs=cnn_kwargs,
        k_folds=5, 
        epochs=15, 
        batch_size=64
    )
    
if __name__ == '__main__':
    main()