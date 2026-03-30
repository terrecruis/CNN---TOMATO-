# 🍅 Tomato Leaf Disease Detection with CNN

Questo progetto implementa una **Rete Neurale Convoluzionale (CNN)** personalizzata sviluppata in **PyTorch** per il riconoscimento automatico delle malattie nelle piante di pomodoro. Il sistema è in grado di classificare le immagini in 10 categorie distinte (9 patologie specifiche e 1 stato sano) utilizzando il dataset *PlantVillage*.

L'obiettivo scientifico dell'esperimento è analizzare come diverse strategie di ottimizzazione (Online, Mini-batch, Full Batch) influenzino l'accuratezza finale e la velocità di convergenza del modello.

---

## 🏗️ Architettura della Rete (TomatoCNN)

Il modello è una CNN progettata per processare immagini di input **64x64x3**. L'architettura è ottimizzata per estrarre feature spaziali complesse mantenendo un carico computazionale gestibile.

### 1. Feature Extraction (Livelli Convoluzionali)
* **Layer 1**: `Conv2d` (3 -> 32 filtri, kernel 3x3) + `ReLU` + `MaxPool2d` (Output: 32x32)
* **Layer 2**: `Conv2d` (32 -> 64 filtri, kernel 3x3) + `ReLU` + `MaxPool2d` (Output: 16x16)
* **Layer 3**: `Conv2d` (64 -> 128 filtri, kernel 3x3) + `ReLU` + `MaxPool2d` (Output: 8x8)

### 2. Classificazione (Livelli Fully Connected)
* **Flattening**: Conversione delle mappe di feature in un vettore 1D da 8192 neuroni.
* **FC 1**: Layer lineare da 512 neuroni con attivazione `ReLU`.
* **Dropout**: Probabilità del 50% per prevenire l'overfitting durante l'addestramento.
* **Output Layer**: 10 neuroni (corrispondenti alle classi del dataset) con attivazione lineare.

---

## 🛠️ Requisiti e Installazione

### 1. Prerequisiti
* **Python 3.10** o superiore.
* Sistema Operativo: Windows (PowerShell o CMD consigliati).

### 2. Setup Ambiente Virtuale
Dalla cartella principale del progetto, esegui i seguenti comandi per isolare le dipendenze:

```powershell
# Creazione dell'ambiente virtuale
python -m venv venv

# Attivazione dell'ambiente (Windows)
.\venv\Scripts\activate
```
Installazione delle dipendenze:
```
pip install torch torchvision scikit-learn matplotlib kagglehub numpy Pillow

