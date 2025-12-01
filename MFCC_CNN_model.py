import os
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

class AudioDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, max_duration=1.0, n_mfcc=40):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.max_len = int(target_sample_rate * max_duration)
        self.n_mfcc = n_mfcc

        # TF para MFCC
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

        # Cargar archivos
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

        waveform, sr = sf.read(path, dtype='float32')
        waveform = torch.tensor(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Padding / truncado
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif waveform.shape[1] < self.max_len:
            padding = self.max_len - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, padding))

        # Normalizar
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        # Extraer MFCCs
        mfcc = self.mfcc_transform(waveform)

        return mfcc, torch.tensor(label, dtype=torch.long)

def save_mfcc_images(dataset, output_dir="mfcc_images", samples_per_class=3):
    os.makedirs(output_dir, exist_ok=True)
    class_names = ["cat", "dog"]
    saved_counts = {0: 0, 1: 0}

    for i in range(len(dataset)):
        mfcc, label = dataset[i]
        label = int(label)

        if saved_counts[label] < samples_per_class:
            plt.figure(figsize=(6, 4))
            plt.imshow(mfcc.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
            plt.title(f"MFCC - {class_names[label]}")
            plt.xlabel("Tiempo (frames)")
            plt.ylabel("Coeficientes MFCC")
            plt.colorbar(label="Magnitud")
            plt.tight_layout()

            filename = os.path.join(output_dir, f"{class_names[label]}_{saved_counts[label] + 1}.png")
            plt.savefig(filename)
            plt.close()

            saved_counts[label] += 1

        if all(v >= samples_per_class for v in saved_counts.values()):
            break

    print(f"Se generaron imágenes MFCC en '{output_dir}'")


class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

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

    base_path = os.path.join(os.getcwd(), 'cats_dogs')

    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    dataset = AudioDataset(train_path)
    save_mfcc_images(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = AudioDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    best_val_acc = 0.0
    best_model_path = "best_audio_model2.pt"

    epochs = 150
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0

        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
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


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Nuevo mejor modelo guardado en Epoch {epoch+1} con Val Acc = {val_acc:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print(f"\nEntrenamiento finalizado. Mejor modelo guardado con Val Acc = {best_val_acc:.4f}")
    print(f"Ruta: {os.path.abspath(best_model_path)}")


    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    test_acc = correct_test / total_test
    print(f"\n Accuracy final en TEST: {test_acc:.4f}")


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label="Train Acc")
    plt.plot(val_acc_history, label="Val Acc")
    plt.title("Accuracy por Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.title("Loss por Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
