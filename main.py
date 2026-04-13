import torch
import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import random

# --- IMPORT VECCHI E NUOVI ---
from dataset import get_dataset, get_train_test_split, get_datasets_kfold
from model import TomatoCNN, TomatoFCNN
from train import train_minibatch, train_stratified_kfold
from analysis import (
    analyze_dataset, 
    plot_dataset_stats, 
    plot_confusion_matrix, 
    plot_overfitting_analysis,
    plot_kfold_results,         # Nuovi grafici K-Fold
    plot_classification_report,  # Nuovi grafici K-Fold
    plot_augmented_images,
    plot_class_grid,
    plot_original_vs_augmented
)

# --- CONFIGURAZIONE GLOBALE ---
EPOCHS = 15
BATCH_SIZE = 64

FCNN_CONFIGS = [
    {'num_hidden_layers': 1, 'hidden_size': 256,  'activation': 'relu',    'label': 'FCNN [1L-256-ReLU]'},
    {'num_hidden_layers': 2, 'hidden_size': 512,  'activation': 'relu',    'label': 'FCNN [2L-512-ReLU]'},
    {'num_hidden_layers': 3, 'hidden_size': 512,  'activation': 'relu',    'label': 'FCNN [3L-512-ReLU]'},
    {'num_hidden_layers': 2, 'hidden_size': 512,  'activation': 'tanh',    'label': 'FCNN [2L-512-Tanh]'},
    {'num_hidden_layers': 2, 'hidden_size': 1024, 'activation': 'relu',    'label': 'FCNN [2L-1024-ReLU]'},
]

CNN_CONFIGS = [
    {'n_filters': 16, 'kernel_size': 3, 'num_blocks': 2, 'label': 'CNN [16f-k3-2blk]'},
    {'n_filters': 32, 'kernel_size': 3, 'num_blocks': 3, 'label': 'CNN [32f-k3-3blk]'},
    {'n_filters': 32, 'kernel_size': 5, 'num_blocks': 3, 'label': 'CNN [32f-k5-3blk]'},
    {'n_filters': 64, 'kernel_size': 3, 'num_blocks': 3, 'label': 'CNN [64f-k3-3blk]'},
    {'n_filters': 32, 'kernel_size': 3, 'num_blocks': 2, 'label': 'CNN [32f-k3-2blk]'},
]

trained_models = {}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed impostato a {seed} per riproducibilità.")

def run_experiment(model, train_data, test_data, label):
    print(f"\n  -> Training: {label}")
    history = train_minibatch(model, train_data, test_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
    final_acc = history['accuracy'][-1]
    print(f"  -> Accuratezza finale: {final_acc:.2f}%")
    trained_models[label] = model

    os.makedirs('weights', exist_ok=True)
    safe_label = label.replace(' ', '_').replace('[', '').replace(']', '')
    torch.save(model.state_dict(), f'weights/{safe_label}.pth')
    return history

def plot_comparison(fcnn_results, cnn_results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('P14 - Confronto FCNN vs CNN su PlantVillage (Pomodoro)', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    for label, history in fcnn_results.items():
        ax1.plot(history['accuracy'], marker='o', markersize=3, label=label)
    ax1.set_title('FCNN - Variazione Iperparametri')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuratezza Test (%)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.4)

    ax2 = axes[1]
    for label, history in cnn_results.items():
        ax2.plot(history['accuracy'], marker='s', markersize=3, label=label)
    ax2.set_title('CNN - Variazione Iperparametri')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Accuratezza Test (%)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.4)

    ax3 = axes[2]
    best_fcnn_label = max(fcnn_results, key=lambda k: fcnn_results[k]['accuracy'][-1])
    best_cnn_label  = max(cnn_results,  key=lambda k: cnn_results[k]['accuracy'][-1])
    ax3.plot(fcnn_results[best_fcnn_label]['accuracy'], 'b-o', markersize=4, label=f'Best FCNN\n{best_fcnn_label}')
    ax3.plot(cnn_results[best_cnn_label]['accuracy'],   'r-s', markersize=4, label=f'Best CNN\n{best_cnn_label}')
    ax3.set_title('Migliore FCNN vs Migliore CNN')
    ax3.set_xlabel('Epoche')
    ax3.set_ylabel('Accuratezza Test (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('confronto_fcnn_vs_cnn.png', dpi=150)
    print("Grafico salvato: confronto_fcnn_vs_cnn.png")

def plot_loss_curves(fcnn_results, cnn_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learning Dynamics - Loss Curves  (— Train  |  -- Validation)',
                 fontsize=13, fontweight='bold')

    # --- FCNN ---
    for label, history in fcnn_results.items():
        line, = axes[0].plot(history['loss'], marker='o', markersize=3, label=label)
        axes[0].plot(history['val_loss'], linestyle='--', color=line.get_color(), alpha=0.6)
    axes[0].set_title('FCNN - Train vs Validation Loss')
    axes[0].set_xlabel('Epoche')
    axes[0].set_ylabel('Loss (Cross-Entropy)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.4)

    # --- CNN ---
    for label, history in cnn_results.items():
        line, = axes[1].plot(history['loss'], marker='s', markersize=3, label=label)
        axes[1].plot(history['val_loss'], linestyle='--', color=line.get_color(), alpha=0.6)
    axes[1].set_title('CNN - Train vs Validation Loss')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Loss (Cross-Entropy)')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    print("Grafico salvato: loss_curves.png")

def print_summary_table(fcnn_results, cnn_results):
    print("\n" + "="*65)
    print(f"{'MODELLO':<30} {'ACC. FINALE':>12} {'EPOCHE A >80%':>14}")
    print("="*65)

    all_results = [('FCNN', fcnn_results), ('CNN', cnn_results)]
    for model_type, results in all_results:
        for label, history in results.items():
            acc = history['accuracy'][-1]
            epochs_to_80 = next((i+1 for i, a in enumerate(history['accuracy']) if a >= 80.0), None)
            epochs_str = str(epochs_to_80) if epochs_to_80 else ">15"
            print(f"  {label:<28} {acc:>10.2f}%  {epochs_str:>12}")
        print("-"*65)

def main():
    set_seed(42)
    print("="*55)
    print("  P14 - FCNN vs CNN su PlantVillage (Completo)")
    print("="*55)

    dataset_path = kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")

    # =====================================================
    # PARTE 1: ANALISI DATASET E CONFRONTO
    # =====================================================
    print("\n[FASE 1] Analisi dataset e confronto modelli base...")
    full_dataset = get_dataset(dataset_path)
    df_classes = analyze_dataset(full_dataset)
    plot_dataset_stats(full_dataset)
    plot_class_grid(full_dataset)
    train_data, test_data = get_train_test_split(full_dataset)

    # Esperimento A: FCNN
    fcnn_results = {}
    num_classes = len(full_dataset.classes)
    for cfg in FCNN_CONFIGS:
        model = TomatoFCNN(cfg['num_hidden_layers'], cfg['hidden_size'], cfg['activation'], num_classes)
        fcnn_results[cfg['label']] = run_experiment(model, train_data, test_data, cfg['label'])

    # Esperimento B: CNN
    cnn_results = {}
    for cfg in CNN_CONFIGS:
        model = TomatoCNN(cfg['n_filters'], cfg['kernel_size'], cfg['num_blocks'], num_classes)
        cnn_results[cfg['label']] = run_experiment(model, train_data, test_data, cfg['label'])

    # Generazione vecchi grafici
    print_summary_table(fcnn_results, cnn_results)
    plot_comparison(fcnn_results, cnn_results)
    plot_loss_curves(fcnn_results, cnn_results)
    plot_overfitting_analysis(fcnn_results, cnn_results)

    best_fcnn_label = max(fcnn_results, key=lambda k: fcnn_results[k]['accuracy'][-1])
    best_cnn_label  = max(cnn_results,  key=lambda k: cnn_results[k]['accuracy'][-1])
    plot_confusion_matrix(trained_models[best_fcnn_label], test_data, full_dataset.classes, model_label=best_fcnn_label)
    plot_confusion_matrix(trained_models[best_cnn_label],  test_data, full_dataset.classes, model_label=best_cnn_label)

    # =====================================================
    # PARTE 2: VALIDAZIONE AVANZATA SUL MIGLIORE
    # =====================================================
    print("\n" + "="*55)
    print("  FASE 2: VALIDAZIONE AVANZATA (K-Fold + Augmentation)")
    print("="*55)
    
    # Ricarichiamo il dataset appositamente per K-Fold
    train_dataset_kf, val_dataset_kf = get_datasets_kfold(dataset_path)
    plot_original_vs_augmented(train_dataset_kf, val_dataset_kf)  
    plot_augmented_images(train_dataset_kf)

    best_cnn_cfg = next(cfg for cfg in CNN_CONFIGS if cfg['label'] == best_cnn_label)
    cnn_kwargs = {
        'n_filters':   best_cnn_cfg['n_filters'],
        'kernel_size': best_cnn_cfg['kernel_size'],
        'num_blocks':  best_cnn_cfg['num_blocks'],
        'num_classes': num_classes
    }
    print(f"\n🏆 CNN selezionata per K-Fold: {best_cnn_label}")
    print(f"   Configurazione: {cnn_kwargs}")
    
    # NOTA: assicurati che la tua funzione train_stratified_kfold ritorni 3 elementi (results, model, report)
    results, kf_model, class_report = train_stratified_kfold(
        train_dataset=train_dataset_kf, 
        val_dataset=val_dataset_kf,
        model_class=TomatoCNN,
        model_kwargs=cnn_kwargs,
        k_folds=5, 
        epochs=15, 
        batch_size=64
    )

    print("\n📊 Generazione grafici K-Fold...")
    plot_kfold_results(results)
    
    class_names = [name.replace('Tomato___', '') for name in train_dataset_kf.classes]
    plot_confusion_matrix(kf_model, val_dataset_kf, class_names, model_label='CNN_KFold_Final')
    plot_classification_report(class_report, class_names)
    
    print("\n✅ PROGETTO P14 COMPLETATO CON SUCCESSO!")

if __name__ == '__main__':
    main()