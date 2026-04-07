# main.py
import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_dataset, get_train_test_split
from model import TomatoCNN, TomatoFCNN
from train import train_minibatch   # usiamo mini-batch (Adam) per tutti gli esperimenti

# --- CONFIGURAZIONE GLOBALE ---
EPOCHS = 15
BATCH_SIZE = 64

# ================================================================
# Configurazioni iperparametri da confrontare (come da traccia)
# ================================================================

# FCNN: varia num_hidden_layers, hidden_size, activation
FCNN_CONFIGS = [
    {'num_hidden_layers': 1, 'hidden_size': 256,  'activation': 'relu',    'label': 'FCNN [1L-256-ReLU]'},
    {'num_hidden_layers': 2, 'hidden_size': 512,  'activation': 'relu',    'label': 'FCNN [2L-512-ReLU]'},
    {'num_hidden_layers': 3, 'hidden_size': 512,  'activation': 'relu',    'label': 'FCNN [3L-512-ReLU]'},
    {'num_hidden_layers': 2, 'hidden_size': 512,  'activation': 'tanh',    'label': 'FCNN [2L-512-Tanh]'},
    {'num_hidden_layers': 2, 'hidden_size': 1024, 'activation': 'relu',    'label': 'FCNN [2L-1024-ReLU]'},
]

# CNN: varia n_filters, kernel_size, num_blocks
CNN_CONFIGS = [
    {'n_filters': 16, 'kernel_size': 3, 'num_blocks': 2, 'label': 'CNN [16f-k3-2blk]'},
    {'n_filters': 32, 'kernel_size': 3, 'num_blocks': 3, 'label': 'CNN [32f-k3-3blk]'},
    {'n_filters': 32, 'kernel_size': 5, 'num_blocks': 3, 'label': 'CNN [32f-k5-3blk]'},
    {'n_filters': 64, 'kernel_size': 3, 'num_blocks': 3, 'label': 'CNN [64f-k3-3blk]'},
    {'n_filters': 32, 'kernel_size': 3, 'num_blocks': 2, 'label': 'CNN [32f-k3-2blk]'},
]


def run_experiment(model, train_data, test_data, label):
    """Wrapper che addestra un modello e restituisce history + accuratezza finale."""
    print(f"\n  -> Training: {label}")
    history = train_minibatch(model, train_data, test_data,
                               batch_size=BATCH_SIZE, epochs=EPOCHS)
    final_acc = history['accuracy'][-1]
    print(f"  -> Accuratezza finale: {final_acc:.2f}%")
    return history


def plot_comparison(fcnn_results, cnn_results):
    """Genera 3 grafici: FCNN, CNN, e confronto migliori."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('P14 - Confronto FCNN vs CNN su PlantVillage (Pomodoro)', fontsize=14, fontweight='bold')

    # --- Grafico 1: Tutte le configurazioni FCNN ---
    ax1 = axes[0]
    for label, history in fcnn_results.items():
        ax1.plot(history['accuracy'], marker='o', markersize=3, label=label)
    ax1.set_title('FCNN - Variazione Iperparametri')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuratezza Test (%)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.4)

    # --- Grafico 2: Tutte le configurazioni CNN ---
    ax2 = axes[1]
    for label, history in cnn_results.items():
        ax2.plot(history['accuracy'], marker='s', markersize=3, label=label)
    ax2.set_title('CNN - Variazione Iperparametri')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Accuratezza Test (%)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.4)

    # --- Grafico 3: Migliore FCNN vs Migliore CNN (+ loss) ---
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
    plt.show()
    print("\nGrafico salvato: confronto_fcnn_vs_cnn.png")


def plot_loss_curves(fcnn_results, cnn_results):
    """Grafico separato per le loss curves (training dynamics)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learning Dynamics - Loss Curves', fontsize=13, fontweight='bold')

    for label, history in fcnn_results.items():
        axes[0].plot(history['loss'], marker='o', markersize=3, label=label)
    axes[0].set_title('FCNN - Training Loss')
    axes[0].set_xlabel('Epoche')
    axes[0].set_ylabel('Loss (Cross-Entropy)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.4)

    for label, history in cnn_results.items():
        axes[1].plot(history['loss'], marker='s', markersize=3, label=label)
    axes[1].set_title('CNN - Training Loss')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Loss (Cross-Entropy)')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    plt.show()
    print("Grafico salvato: loss_curves.png")


def print_summary_table(fcnn_results, cnn_results):
    """Stampa tabella riassuntiva degli esperimenti."""
    print("\n" + "="*65)
    print(f"{'MODELLO':<30} {'ACC. FINALE':>12} {'EPOCHE A >80%':>14}")
    print("="*65)

    all_results = [('FCNN', fcnn_results), ('CNN', cnn_results)]
    for model_type, results in all_results:
        for label, history in results.items():
            acc = history['accuracy'][-1]
            # epoche necessarie per superare 80%
            epochs_to_80 = next((i+1 for i, a in enumerate(history['accuracy']) if a >= 80.0), None)
            epochs_str = str(epochs_to_80) if epochs_to_80 else ">15"
            print(f"  {label:<28} {acc:>10.2f}%  {epochs_str:>12}")
        print("-"*65)

    best_fcnn = max(fcnn_results, key=lambda k: fcnn_results[k]['accuracy'][-1])
    best_cnn  = max(cnn_results,  key=lambda k: cnn_results[k]['accuracy'][-1])
    print(f"\n  Migliore FCNN: {best_fcnn}  ({fcnn_results[best_fcnn]['accuracy'][-1]:.2f}%)")
    print(f"  Migliore CNN:  {best_cnn}  ({cnn_results[best_cnn]['accuracy'][-1]:.2f}%)")
    print("="*65)


def main():
    print("="*55)
    print("  P14 - FCNN vs CNN su PlantVillage (Malattie Pomodoro)")
    print("="*55)

    # 1. Download dataset
    print("\nScaricamento/Verifica dataset...")
    dataset_path = kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")
    print(f"Path: {dataset_path}")

    # 2. Caricamento
    from dataset import get_dataset, get_train_test_split
    full_dataset = get_dataset(dataset_path)
    print(f"Dataset: {len(full_dataset)} immagini | {len(full_dataset.classes)} classi")
    print(f"Classi: {full_dataset.classes}")

    train_data, test_data = get_train_test_split(full_dataset)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # =====================================================
    # ESPERIMENTO A: FCNN con diversi iperparametri
    # =====================================================
    print("\n" + "="*55)
    print("  ESPERIMENTO A: FCNN (Fully Connected)")
    print("="*55)

    fcnn_results = {}
    num_classes = len(full_dataset.classes)

    for cfg in FCNN_CONFIGS:
        label = cfg['label']
        model = TomatoFCNN(
            num_hidden_layers=cfg['num_hidden_layers'],
            hidden_size=cfg['hidden_size'],
            activation=cfg['activation'],
            num_classes=num_classes
        )
        fcnn_results[label] = run_experiment(model, train_data, test_data, label)

    # =====================================================
    # ESPERIMENTO B: CNN con diversi iperparametri
    # =====================================================
    print("\n" + "="*55)
    print("  ESPERIMENTO B: CNN (Convoluzionale)")
    print("="*55)

    cnn_results = {}

    for cfg in CNN_CONFIGS:
        label = cfg['label']
        model = TomatoCNN(
            n_filters=cfg['n_filters'],
            kernel_size=cfg['kernel_size'],
            num_blocks=cfg['num_blocks'],
            num_classes=num_classes
        )
        cnn_results[label] = run_experiment(model, train_data, test_data, label)

    # =====================================================
    # RISULTATI E GRAFICI
    # =====================================================
    print_summary_table(fcnn_results, cnn_results)
    plot_comparison(fcnn_results, cnn_results)
    plot_loss_curves(fcnn_results, cnn_results)
    print("\nEsperimento completato!")


if __name__ == '__main__':
    main()