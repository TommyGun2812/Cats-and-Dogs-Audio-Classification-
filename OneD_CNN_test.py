import argparse
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import os
from torchaudio.transforms import Resample


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


def preprocess_audio(path, target_sample_rate=16000, max_duration=1.0):
    waveform, sr = sf.read(path, dtype='float32')

    # Convertir a tensor
    waveform = torch.tensor(waveform)

    # Asegurar forma [1, N]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.mean(dim=1, keepdim=True)  # Convertir a mono

    # Resample
    if sr != target_sample_rate:
        resampler = Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # Padding / truncamiento
    max_len = int(target_sample_rate * max_duration)

    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    elif waveform.shape[1] < max_len:
        padding = max_len - waveform.shape[1]
        waveform = nn.functional.pad(waveform, (0, padding))

    return waveform.unsqueeze(0)   # [1, 1, N]


def predict(audio_path, model_path="best_audio_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = AudioClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocesar audio
    waveform = preprocess_audio(audio_path)
    waveform = waveform.to(device)

    # Inferencia
    with torch.no_grad():
        outputs = model(waveform)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    class_names = ["Cat", "Dog"]
    pred_label = class_names[pred_class]
    confidence = probs[0, pred_class].item() * 100

    print(f"\n Archivo: {os.path.basename(audio_path)}")
    print(f"Predicción: {pred_label}")
    print(f"Confianza: {confidence:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificación de audio (Cat vs Dog)")
    parser.add_argument("--audio_path", type=str, required=True, help="Ruta del archivo .wav a predecir")
    parser.add_argument("--model_path", type=str, default="best_audio_model1.pt", help="Ruta del modelo entrenado")
    args = parser.parse_args()

    predict(args.audio_path, args.model_path)
