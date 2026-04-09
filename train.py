import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm  # Libreria per la barra di caricamento

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
        # tqdm qui è vitale perché le iterazioni sono pari al numero di immagini!
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
    """Esegue la K-Fold Cross Validation con barre di caricamento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\n' + "="*40)
        print(f' FOLD {fold+1}/{k_folds}')
        print("="*40)
        
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        
        # Determina il numero di classi dinamicamente dal dataset
        num_classes = len(dataset.classes)
        model = model_class(n_filters=n_filters, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100.0 * correct / total
        results[fold] = accuracy
        print(f'-> Accuratezza Fold {fold+1}: {accuracy:.2f}%')
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\n' + "#"*40)
    print(f' MEDIA ACCURATEZZA {k_folds}-FOLD: {np.mean(list(results.values())):.2f}%')
    print("#"*40)
    return results