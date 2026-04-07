#!/usr/bin/env python3
"""
Analisi completa dataset PlantVillage Tomato (richiesta prof).
Da importare in main.py.
"""

from collections import Counter
import pandas as pd
import numpy as np

def analyze_dataset(full_dataset, output_dir='.', verbose=True):
    """
    Analizza distribuzione classi e salva report.
    
    Args:
        full_dataset: torchvision ImageFolder dataset
        output_dir: cartella per CSV output
        verbose: stampa dettagli console
    
    Returns:
        pd.DataFrame con distribuzione classi
    """
    # Distribuzione
    targets = [full_dataset.targets[i] for i in range(len(full_dataset))]
    class_dist = Counter(targets)
    total_images = len(full_dataset)
    
    if verbose:
        print("\n" + "="*70)
        print("📊 ANALISI DATASET PLANTVILLAGE TOMATO")
        print("="*70)
        
        print(f"📈 STATISTICHE GENERALI:")
        print(f"  Totale immagini: {total_images:,}")
        print(f"  Numero classi: {len(full_dataset.classes)}")
        
        print(f"\n📋 DISTRIBUZIONE PER CLASSE:")
        print(f"{'Classe':<35} {'Immagini':>8} {'%':>6}")
        print("-" * 50)
        
        for idx, cls in enumerate(full_dataset.classes):
            count = class_dist[idx]
            pct = 100 * count / total_images
            print(f"{cls:<35} {count:>8,} {pct:>6.1f}%")
        
        # Bilanciamento
        mean_per_class = total_images / len(full_dataset.classes)
        std_per_class = np.std(list(class_dist.values()))
        cv = std_per_class / mean_per_class
        print(f"\n⚖️  BILANCIAMENTO:")
        print(f"  Media/classe: {mean_per_class:.0f}")
        print(f"  Dev.std: {std_per_class:.0f}")
        print(f"  CV: {cv:.2f} {'(moderato sbilanciamento)' if cv > 0.3 else '(bilanciato)'}")
    
    # DataFrame per PPT/relazione
    df_classes = pd.DataFrame([
        {
            'Classe': cls, 
            'Immagini': class_dist[i], 
            'Percentuale': f"{100*class_dist[i]/total_images:.1f}%"
        }
        for i, cls in enumerate(full_dataset.classes)
    ])
    
    # Salva CSV
    df_classes.to_csv(f'{output_dir}/dataset_classes.csv', index=False)
    
    if verbose:
        print(f"\n💾 Report salvato: dataset_classes.csv")
    
    return df_classes

if __name__ == '__main__':
    # Test standalone
    import kagglehub
    from dataset import get_dataset
    
    dataset_path = kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")
    ds = get_dataset(dataset_path)
    analyze_dataset(ds)