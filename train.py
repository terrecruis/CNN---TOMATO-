import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np

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
        # Nell'online learning, questo ciclo itera decine di migliaia di volte
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = 100 * evaluate_model(model, test_loader, device) / len(test_data)
        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        
        print(f"[Online] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    
    return history


def train_minibatch(model, train_data, test_data, batch_size=64, epochs=10, lr=0.001):
    """MINI-BATCH LEARNING: Aggiorna i pesi ogni N immagini (es. 64). Compromesso ideale."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = 100 * evaluate_model(model, test_loader, device) / len(test_data)
        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        
        print(f"[Mini-Batch] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    
    return history


def train_batch(model, train_data, test_data, epochs=10, lr=0.001):
    """BATCH LEARNING: Aggiorna i pesi UNA SOLA volta per epoca guardando tutto il dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Il batch_size è uguale all'intera lunghezza del train_data
    full_batch_size = len(train_data)
    train_loader = DataLoader(train_data, batch_size=full_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Questo ciclo eseguirà UNA SOLA iterazione per epoca
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = 100 * evaluate_model(model, test_loader, device) / len(test_data)
        
        # Non serve dividere per len(train_loader) perché è sempre 1
        history['loss'].append(running_loss) 
        history['accuracy'].append(acc)
        
        print(f"[Full Batch] Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Acc: {acc:.2f}%")
    
    return history

def train_kfold(dataset, model_class, k_folds=5, epochs=5, batch_size=64, n_filters=32):
    """Esegue la K-Fold Cross Validation sull'intero dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\n--- FOLD {fold+1}/{k_folds} ---')
        
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        
        model = model_class(n_filters=n_filters).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        # Valutazione del fold
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
        print(f'Accuratezza per il fold {fold+1}: {accuracy:.2f}%')
        
        # Pulizia memoria GPU tra un fold e l'altro
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\nMedia Accuratezza {k_folds}-Fold: {np.mean(list(results.values())):.2f}%')
    return results