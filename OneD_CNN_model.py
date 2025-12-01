import os
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import Resample

# Dataset personalizado
class AudioDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, max_duration=1.0, transform=None):
        self.root_dir = root_dir
        self.samples = []
        self.target_sample_rate = target_sample_rate
        self.max_len = int(target_sample_rate * max_duration)
        self.transform = transform

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

        import soundfile as sf
        waveform, sr = sf.read(path, dtype='float32')
        waveform = torch.tensor(waveform).unsqueeze(0) if waveform.ndim == 1 else torch.tensor(waveform.T)

        # Convertir a mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != self.target_sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Padding / truncamiento
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif waveform.shape[1] < self.max_len:
            padding = self.max_len - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, padding))

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, torch.tensor(label, dtype=torch.long)


# Modelo CNN 1D
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Rutas
    base_path = os.path.join(os.getcwd(), 'cats_dogs')

    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    # Dataset + split
    full_dataset = AudioDataset(root_dir=train_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_dataset = AudioDataset(root_dir=test_path)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Dispositivo, modelo, optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Para guardar el mejor modelo
    best_model_path = "best_audio_model.pt"
    best_val_acc = 0.0

    # Historial
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    epochs = 150
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0

        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        train_loss = running_loss / total_train
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # VALIDACIÓN
        model.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)

                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_loss = val_running_loss / total_val
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        # GUARDAR MEJOR MODELO
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Nuevo mejor modelo guardado (Val Acc = {val_acc:.4f})")

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            _, preds = torch.max(outputs, 1)

            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    test_acc = correct_test / total_test
    print(f"\n Accuracy final en TEST: {test_acc:.4f}")


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
