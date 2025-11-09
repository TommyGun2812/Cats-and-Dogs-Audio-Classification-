import os
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# Dataset personalizado para audio
class AudioDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, max_duration=1.0, n_mfcc=40):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.max_len = int(target_sample_rate * max_duration)
        self.n_mfcc = n_mfcc

        # Transformación para extraer características MFCC
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

        # Cargar muestras de audio (gatos y perros)
        self.samples = []
        for label, cls in enumerate(["cat", "dog"]):
            folder = os.path.join(root_dir, cls)
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    self.samples.append({
                        "path": os.path.join(folder, file),
                        "label": label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path, label = sample["path"], sample["label"]

        # Leer archivo de audio
        waveform, sr = sf.read(path, dtype='float32')
        waveform = torch.tensor(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resamplear si es necesario
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Padding o recorte a longitud fija
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif waveform.shape[1] < self.max_len:
            padding = self.max_len - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, padding))

        # Normalización
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        # Extraer características MFCC
        mfcc = self.mfcc_transform(waveform)

        return mfcc, torch.tensor(label, dtype=torch.long)

# Función para visualizar MFCCs
def save_mfcc_images(dataset, output_dir="mfcc_images", samples_per_class=3):
    os.makedirs(output_dir, exist_ok=True)
    class_names = ["cat", "dog"]
    saved_counts = {0: 0, 1: 0}

    for i in range(len(dataset)):
        mfcc, label = dataset[i]
        cls_name = class_names[label]

        label = int(label)
        if saved_counts[label] < samples_per_class:
            plt.figure(figsize=(6, 4))
            plt.imshow(mfcc.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
            plt.title(f"MFCC - {cls_name}")
            plt.xlabel("Tiempo (frames)")
            plt.ylabel("Coeficientes MFCC")
            plt.colorbar(label="Magnitud")
            plt.tight_layout()

            filename = os.path.join(output_dir, f"{cls_name}_{saved_counts[label] + 1}.png")
            plt.savefig(filename)
            plt.close()

            saved_counts[label] += 1

        if all(v >= samples_per_class for v in saved_counts.values()):
            break

    print(f"Se generaron y guardaron imágenes MFCC en la carpeta '{output_dir}'")

# Modelo de red neuronal convolucional
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        
        # Capas convolucionales
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Capas densas
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Ruta del dataset de entrenamiento
    train_path = r"C:\Users\perez\OneDrive\Documentos\Concentracion IA 2\DeepLearning\Reto\cats_dogs\train"

    # Crear dataset y generar imágenes MFCC
    dataset = AudioDataset(train_path)
    save_mfcc_images(dataset)

    # Dividir dataset en entrenamiento y validación
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders para entrenamiento
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Configurar dispositivo (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Historial para métricas
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    # Entrenamiento con early stopping implícito
    best_val_loss = float("inf")
    best_model_path = "best_audio_model.pt"

    epochs = 150
    for epoch in range(epochs):
        # Fase de entrenamiento
        model.train()
        correct_train, total_train, running_loss = 0, 0, 0.0
        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calcular métricas
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        train_loss = running_loss / total_train
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # Fase de validación
        model.eval()
        correct_val, total_val, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for mfccs, labels in val_loader:
                mfccs, labels = mfccs.to(device), labels.to(device)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_loss = val_running_loss / total_val
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Nuevo mejor modelo guardado (Epoch {epoch+1}) con Val Loss: {val_loss:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print(f"Entrenamiento finalizado. Mejor modelo guardado con Val Loss = {best_val_loss:.4f}")
    print(f"Ruta del mejor modelo: {os.path.abspath(best_model_path)}")

    # Graficar resultados
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
