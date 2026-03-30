# main.py
import os
import kagglehub
import matplotlib.pyplot as plt
from dataset import get_dataset, get_train_test_split
from model import TomatoCNN
from train import train_kfold, train_online, train_minibatch, train_batch

# --- CONFIGURAZIONE ---
N_FILTERS = 32
EPOCHS = 10

def plot_results(history_dict):
    """Genera il grafico comparativo delle accuratezze."""
    plt.figure(figsize=(10, 5))
    for name, history in history_dict.items():
        if history['accuracy']: 
            plt.plot(history['accuracy'], label=name, marker='o')
        
    plt.title('Confronto Modalità di Apprendimento (Accuracy)')
    plt.xlabel('Epoche')
    plt.ylabel('Accuratezza (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('confronto_modalita.png')
    plt.show()

def main():
    print("="*50)
    print("PREPARAZIONE DATASET")
    print("="*50)
    
    # 1. Download o recupero del dataset dalla cache tramite kagglehub
    print("Scaricamento/Verifica del dataset tramite kagglehub in corso...")
    dataset_path = kagglehub.dataset_download("charuchaudhry/plantvillage-tomato-leaf-dataset")
    print(f"Path to dataset files: {dataset_path}")

    # A seconda di come è strutturato lo zip su Kaggle, le cartelle delle classi 
    # potrebbero trovarsi direttamente in 'dataset_path' o dentro una sottocartella.
    # Proviamo ad usare direttamente il path restituito.
    data_dir = dataset_path

    # 2. Caricamento in PyTorch tramite dataset.py
    print("\nCaricamento dataset in PyTorch...")
    try:
        full_dataset = get_dataset(data_dir)
        print(f"Dataset caricato con successo: {len(full_dataset)} immagini totali.")
        print(f"Numero di classi trovate: {len(full_dataset.classes)}")
    except Exception as e:
        print(f"\n[!] Errore durante il caricamento del dataset: {e}")
        print("Suggerimento: Controlla se le cartelle delle classi si trovano in una sottocartella del path restituito.")
        return

    # ==========================================
    # ESPERIMENTO 1: K-FOLD CROSS VALIDATION
    # ==========================================
    print("\n" + "="*50)
    print("AVVIO ESPERIMENTO 1: K-FOLD CROSS VALIDATION")
    print("="*50)
    train_kfold(full_dataset, TomatoCNN, k_folds=5, epochs=5, batch_size=64, n_filters=N_FILTERS)

    # ==========================================
    # ESPERIMENTO 2: CONFRONTO MODALITÀ (Online, Mini, Batch)
    # ==========================================
    print("\n" + "="*50)
    print("AVVIO ESPERIMENTO 2: STUDIO DEL BATCH SIZE")
    print("="*50)
    
    train_data, test_data = get_train_test_split(full_dataset)
    histories = {}
    
    # 1. Online Learning (Batch Size = 1)
    print("\n---> Modalità: Online Learning (Aggiornamento per singola immagine)")
    model_online = TomatoCNN(n_filters=N_FILTERS)
    histories['Online (BS=1)'] = train_online(model_online, train_data, test_data, epochs=3)

    # 2. Mini-batch Learning (Batch Size = 64)
    print("\n---> Modalità: Mini-batch Learning (Aggiornamento ogni 64 immagini)")
    model_mini = TomatoCNN(n_filters=N_FILTERS)
    histories['Mini-batch (BS=64)'] = train_minibatch(model_mini, train_data, test_data, batch_size=64, epochs=EPOCHS)

    # 3. Batch Learning (Full Dataset)
    print("\n---> Modalità: Batch Learning (Aggiornamento sull'intero dataset)")
    model_batch = TomatoCNN(n_filters=N_FILTERS)
    histories['Batch (Full)'] = {'loss': [], 'accuracy': []}
    try:
        histories['Batch (Full)'] = train_batch(model_batch, train_data, test_data, epochs=EPOCHS)
    except RuntimeError as e:
        print(f"\n[!] Errore di memoria rilevato: {e}")
        print("[!] Il Batch Learning ha saturato la VRAM. Il grafico verrà generato senza questa curva.")

    # Generazione Grafico Finale
    print("\nGenerazione del grafico comparativo in corso...")
    plot_results(histories)
    print("Processo completato! Il grafico è stato salvato come 'confronto_modalita.png'.")

if __name__ == '__main__':
    main()