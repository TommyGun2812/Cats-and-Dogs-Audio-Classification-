import argparse
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import os

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
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


def preprocess_audio(path, target_sample_rate=16000, max_duration=1.0, n_mfcc=40):
    # Leer archivo
    waveform, sr = sf.read(path, dtype='float32')
    waveform = torch.tensor(waveform)

    # Asegurar que sea [1, N]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Resamplear si es necesario
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # Padding o recorte
    max_len = int(target_sample_rate * max_duration)
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    elif waveform.shape[1] < max_len:
        padding = max_len - waveform.shape[1]
        waveform = nn.functional.pad(waveform, (0, padding))

    # Normalización
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

    # Extraer MFCCs
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=target_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
    )
    mfcc = mfcc_transform(waveform)

    return mfcc.unsqueeze(0)  # [1, 1, n_mfcc, tiempo]

def predict(audio_path, model_path="best_audio_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = AudioClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocesar audio
    mfcc = preprocess_audio(audio_path)
    mfcc = mfcc.to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(mfcc)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    class_names = ["Cat", "Dog"]
    pred_label = class_names[pred_class]
    confidence = probs[0, pred_class].item() * 100

    print(f"\n Archivo: {os.path.basename(audio_path)}")
    print(f"Predicción: {pred_label} ({confidence:.2f}% de confianza)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificación de audio: gato o perro")
    parser.add_argument("--audio_path", type=str, required=True, help="Ruta del archivo .wav a predecir")
    parser.add_argument("--model_path", type=str, default="best_audio_model.pt", help="Ruta del modelo entrenado (.pt)")
    args = parser.parse_args()

    predict(args.audio_path, args.model_path)
