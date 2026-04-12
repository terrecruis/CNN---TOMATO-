"""
Analisi completa dataset PlantVillage Tomato (richiesta prof).
Da importare in main.py.
"""

from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def plot_dataset_stats(full_dataset, output_dir='.'):
    """
    Genera grafici statistici sul dataset PlantVillage Tomato.
    Salva: dataset_stats.png
    """
    from collections import Counter
    
    targets = full_dataset.targets
    classes = full_dataset.classes
    class_dist = Counter(targets)
    
    counts = [class_dist[i] for i in range(len(classes))]
    # Nomi brevi per il grafico
    short_names = [c.replace('Tomato___', '').replace('_', ' ') for c in classes]
    total = sum(counts)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('📊 Analisi Dataset PlantVillage - Tomato', fontsize=14, fontweight='bold')
    
    # --- Grafico 1: Barplot distribuzione classi ---
    ax1 = axes[0]
    colors = plt.cm.Set3.colors[:len(classes)]
    bars = ax1.barh(short_names, counts, color=colors, edgecolor='white', linewidth=0.8)
    ax1.set_xlabel('Numero Immagini')
    ax1.set_title('Distribuzione per Classe')
    ax1.grid(axis='x', alpha=0.3)
    
    # Etichette valori sulle barre
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 f'{count:,} ({100*count/total:.1f}%)',
                 va='center', fontsize=8)
    
    ax1.set_xlim(0, max(counts) * 1.25)
    
    # --- Grafico 2: Pie chart ---
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=None,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax2.set_title('Distribuzione Percentuale')
    ax2.legend(wedges, short_names,
               loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=2, fontsize=8)
    
    plt.tight_layout()
    out_path = f'{output_dir}/dataset_stats.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"💾 Grafico salvato: {out_path}")
    
    return fig

def plot_confusion_matrix(model, test_data, class_names, output_dir='.', model_label='Model'):
    """
    Genera e salva la matrice di confusione per un modello.
    
    Args:
        model: modello PyTorch già addestrato
        test_data: test dataset
        class_names: lista nomi classi
        output_dir: cartella output
        model_label: nome del modello per il titolo
    """
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    loader = DataLoader(test_data, batch_size=64, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcola matrice di confusione manualmente
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # Normalizza per riga (recall per classe)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Nomi brevi per leggibilità
    short_names = [c.replace('Tomato___', '').replace('_', ' ') for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Matrice di Confusione — {model_label}', fontsize=14, fontweight='bold')

    # --- Sinistra: valori assoluti ---
    ax1 = axes[0]
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Valori Assoluti')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_xlabel('Predetto')
    ax1.set_ylabel('Reale')
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=7,
                     color='white' if cm[i, j] > thresh else 'black')

    # --- Destra: normalizzata (recall) ---
    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Normalizzata (Recall per Classe)')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_xlabel('Predetto')
    ax2.set_ylabel('Reale')
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f'{cm_norm[i, j]:.2f}',
                     ha='center', va='center', fontsize=7,
                     color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    safe_label = model_label.replace(' ', '_').replace('[', '').replace(']', '')
    out_path = f'{output_dir}/confusion_matrix_{safe_label}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"💾 Matrice di confusione salvata: {out_path}")

def plot_overfitting_analysis(fcnn_results, cnn_results, output_dir='.'):
    """
    Confronta training loss vs validation loss per ogni modello.
    Evidenzia il punto di overfitting (divergenza delle curve).
    """
    all_configs = list(fcnn_results.items()) + list(cnn_results.items())
    n = len(all_configs)
    cols = 5
    rows = (n + cols - 1) // cols  # 2 righe da 5

    fig, axes = plt.subplots(rows, cols, figsize=(22, 8))
    fig.suptitle('Analisi Overfitting — Training Loss vs Validation Loss', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, (label, history) in enumerate(all_configs):
        ax = axes[i]
        epochs = range(1, len(history['loss']) + 1)

        ax.plot(epochs, history['loss'],     'b-o', markersize=3, label='Train Loss')
        ax.plot(epochs, history['val_loss'], 'r-s', markersize=3, label='Val Loss')

        # Evidenzia il minimo della validation loss (good point to stop)
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.annotate(f'Best\nep.{best_epoch}',
                    xy=(best_epoch, min(history['val_loss'])),
                    xytext=(best_epoch + 0.5, min(history['val_loss']) + 0.05),
                    fontsize=7, color='green')

        # Colora area di overfitting (dove val_loss > train_loss e divergono)
        train_arr = history['loss']
        val_arr   = history['val_loss']
        ep_list   = list(epochs)
        ax.fill_between(ep_list, train_arr, val_arr,
                        where=[v > t for v, t in zip(val_arr, train_arr)],
                        alpha=0.12, color='red', label='Gap (overfit)')

        is_fcnn = label.startswith('FCNN')
        ax.set_title(label, fontsize=8, color='blue' if is_fcnn else 'darkred')
        ax.set_xlabel('Epoche', fontsize=7)
        ax.set_ylabel('Loss', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = f'{output_dir}/overfitting_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"💾 Grafico overfitting salvato: {out_path}")

def plot_kfold_results(kfold_results, output_dir='.'):
    """
    Genera un bar plot delle accuracy dei 5 fold con la linea della media.
    """
    folds = list(kfold_results.keys())
    accuracies = list(kfold_results.values())
    mean_acc = np.mean(accuracies)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Risultati Stratified 5-Fold Cross Validation', fontsize=14, fontweight='bold')

    bars = ax.bar(folds, accuracies, color='skyblue', edgecolor='black')
    
    # Linea della media
    ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Media: {mean_acc:.2f}%')
    
    # Testo sopra le barre
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(min(accuracies) - 5, 100)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f+1}" for f in folds])
    ax.set_ylabel('Accuratezza (%)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = f'{output_dir}/kfold_results.png'
    plt.savefig(out_path, dpi=150)
    print(f"💾 Grafico K-Fold salvato: {out_path}")
    plt.close()


def plot_classification_report(report_text, class_names, output_dir='.'):
    """
    Estrae le metriche dal report testuale generato da sklearn e crea una heatmap.
    """
    lines = report_text.split('\n')
    data = []
    
    # Parse del testo
    for line in lines[2:-5]: 
        row_data = line.split()
        if len(row_data) >= 4:
            # uniamo il nome della classe se ha spazi
            class_name = " ".join(row_data[:-4])
            precision = float(row_data[-4])
            recall = float(row_data[-3])
            f1 = float(row_data[-2])
            data.append([precision, recall, f1])

    data = np.array(data)
    short_names = [c.replace('Tomato___', '').replace('_', ' ') for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='YlGn', vmin=0.5, vmax=1.0)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(short_names)))
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.set_yticklabels(short_names)

    # Inserisci i valori nelle celle
    for i in range(len(short_names)):
        for j in range(3):
            text_color = "white" if data[i, j] > 0.85 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=text_color)

    ax.set_title("Classification Report (Ultimo Fold)", fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = f'{output_dir}/classification_report.png'
    plt.savefig(out_path, dpi=150)
    print(f"💾 Grafico Classification Report salvato: {out_path}")
    plt.close()

def plot_augmented_images(dataset, num_images=5, output_dir='.'):
    """
    Visualizza un campione casuale di immagini dal dataset aumentato per verificare le trasformazioni.
    Applica la de-normalizzazione per visualizzare i colori corretti.
    """
    import random

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3.5))
    fig.suptitle('Esempi di Data Augmentation (Train Set K-Fold)', fontsize=14, fontweight='bold')

    # Valori di normalizzazione usati nel dataset
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Prendi 5 immagini a caso
    indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(indices):
        img_tensor, label_idx = dataset[idx]
        
        # Converti da (Canali, Altezza, Larghezza) a (Altezza, Larghezza, Canali)
        img_np = img_tensor.numpy().transpose((1, 2, 0))
        
        # De-normalizza: img = img * std + mean
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)  # Assicura che i valori restino tra 0 e 1
        
        axes[i].imshow(img_np)
        
        # Nome della classe (pulito)
        class_name = dataset.classes[label_idx].replace('Tomato___', '').replace('_', ' ')
        axes[i].set_title(class_name, fontsize=9)
        axes[i].axis('off')
        
    plt.tight_layout()
    out_path = f'{output_dir}/augmented_samples.png'
    plt.savefig(out_path, dpi=150)
    print(f"💾 Esempi di augmentation salvati: {out_path}")
    plt.close()

if __name__ == '__main__':
    # Test standalone
    import kagglehub
    from dataset import get_dataset
    
    dataset_path = kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")
    ds = get_dataset(dataset_path)
    analyze_dataset(ds)
    plot_dataset_stats(ds)
    plot_confusion_matrix(model=None, test_data=ds, class_names=ds.classes, model_label='Test')