import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight  # <-- Aggiunto per i pesi
import numpy as np
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    """Funzione di supporto per calcolare l'accuratezza sul test set."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
    return correct

def train_online(model, train_data, test_data, epochs=3, lr=0.001):
    """ONLINE LEARNING: Aggiorna i pesi dopo OGNI singola immagine (batch_size=1)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64) 

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Online] Epoch {epoch+1}/{epochs}", leave=True)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        acc = 100 * evaluate_model(model, test_loader, device) / len(test_data)
        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        print(f" -> Accuracy: {acc:.2f}%")
    
    return history

def train_minibatch(model, train_data, test_data, batch_size=64, epochs=10, lr=0.001):
    """MINI-BATCH LEARNING: Compromesso ideale."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=64)

    history = {'loss': [], 'val_loss': [], 'accuracy': []}

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Mini-Batch] Epoch {epoch+1}/{epochs}", leave=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION LOSS ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        acc = 100 * correct / len(test_data)

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(acc)
        print(f" -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {acc:.2f}%")

    return history

def train_batch(model, train_data, test_data, epochs=10, lr=0.001):
    """BATCH LEARNING: Aggiorna i pesi una volta per epoca."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    full_batch_size = len(train_data)
    train_loader = DataLoader(train_data, batch_size=full_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"[Full Batch] Epoch {epoch+1}/{epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()

        acc = 100 * evaluate_model(model, test_loader, device) / len(test_data)
        history['loss'].append(running_loss) 
        history['accuracy'].append(acc)
        print(f" -> Accuracy: {acc:.2f}%")
    
    return history

def train_kfold(dataset, model_class, k_folds=5, epochs=5, batch_size=64, n_filters=32):
    # Ometto il corpo per brevità, tanto stiamo usando la Stratified, ma lascialo pure com'era se vuoi.
    pass

def train_stratified_kfold(train_dataset, val_dataset, model_class, model_kwargs, k_folds=5, epochs=15, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = {}
    targets = train_dataset.targets # Necessari per stratificare
    
    # =========================================================================
    # 🎯 NOVITÀ: CALCOLO DEI PESI DELLE CLASSI PER BILANCIARE IL DATASET
    # =========================================================================
    targets_np = np.array(targets)
    classes_np = np.unique(targets_np)
    
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes_np, 
        y=targets_np
    )
    # Convertiamo l'array numpy in un tensore PyTorch e lo mandiamo su GPU/CPU
    pesi_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print("\n⚖️ Pesi applicati alla Loss per bilanciare le classi:")
    print([f"Classe {i}: {w:.2f}" for i, w in enumerate(class_weights)])
    # =========================================================================

    all_true_labels_last_fold = []
    all_pred_labels_last_fold = []

    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f'\n' + "="*40)
        print(f' FOLD {fold+1}/{k_folds}')
        print("="*40)
        
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, sampler=val_subsampler)
        
        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 🎯 PASSIAMO I PESI ALLA FUNZIONE DI LOSS QUI!
        criterion = nn.CrossEntropyLoss(weight=pesi_tensor)
        
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} - Ep {epoch+1}/{epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        # Valutazione
        model.eval()
        correct, total = 0, 0
        fold_true, fold_pred = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Salviamo per il report del fold finale
                if fold == k_folds - 1:
                    fold_true.extend(labels.cpu().numpy())
                    fold_pred.extend(predicted.cpu().numpy())
                
        accuracy = 100.0 * correct / total
        results[fold] = accuracy
        print(f'-> Accuratezza Fold {fold+1}: {accuracy:.2f}%')
        
        # Salviamo i risultati dell'ultimo fold per il Classification Report
        if fold == k_folds - 1:
            all_true_labels_last_fold = fold_true
            all_pred_labels_last_fold = fold_pred

    print(f'\n' + "#"*40)
    print(f' MEDIA ACCURATEZZA {k_folds}-FOLD: {np.mean(list(results.values())):.2f}%')
    print("#"*40)
    
    print("\n--- CLASSIFICATION REPORT (Ultimo Fold) ---")
    class_names = [name.replace('Tomato___', '') for name in train_dataset.classes]
    report_text = classification_report(all_true_labels_last_fold, all_pred_labels_last_fold, target_names=class_names)
    print(report_text)

    return results, model, report_text