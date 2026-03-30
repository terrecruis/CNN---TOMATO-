# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TomatoCNN(nn.Module):
    def __init__(self, n_filters=32, num_classes=10):
        super(TomatoCNN, self).__init__()
        
        # --- BLOCCHI CONVOLUZIONALI (Estrazione delle Feature) ---
        
        # Primo blocco: l'input ha 3 canali (immagini RGB)
        # padding=1 mantiene la dimensione spaziale prima del pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=3, padding=1)
        
        # Layer di Pooling condiviso: dimezza altezza e larghezza dell'immagine
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Secondo blocco: raddoppiamo i filtri per catturare pattern più complessi (texture delle foglie)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=3, padding=1)
        
        # Terzo blocco: quadruplichiamo i filtri iniziali per forme specifiche delle malattie
        self.conv3 = nn.Conv2d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=3, padding=1)
        
        # --- BLOCCHI FULLY CONNECTED (Classificazione) ---
        
        # Calcolo dinamico dei neuroni in ingresso al layer lineare.
        # Partendo da un'immagine 64x64 e applicando MaxPool2d 3 volte (dimezza ogni volta):
        # 64 -> 32 -> 16 -> 8. La dimensione spaziale finale sarà 8x8.
        self.fc_input_dim = (n_filters * 4) * 8 * 8
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        # Il Dropout spegne casualmente il 50% dei neuroni durante il training per evitare l'overfitting
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, num_classes) # Output finale: 10 neuroni (le tue 9 malattie + 1 sano)

    def forward(self, x):
        # Passaggio tensore attraverso le convoluzioni, attivazione ReLU e Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flattening: trasforma la mappa 3D delle feature in un vettore 1D
        # Il '-1' dice a PyTorch di mantenere inalterata la dimensione del Batch Size
        x = x.view(-1, self.fc_input_dim)
        
        # Passaggio attraverso i layer di classificazione
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Nota: Non applichiamo Softmax qui alla fine perché PyTorch 
        # la include già automaticamente all'interno di nn.CrossEntropyLoss() in fase di training.
        return x