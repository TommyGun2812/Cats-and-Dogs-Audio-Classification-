# Clasificación de Audio de Gatos y Perros

Este proyecto implementa y compara dos enfoques de aprendizaje profundo para la clasificación de clips de audio de gatos y perros.  
Los modelos fueron desarrollados con PyTorch y probados mediante un script que permite subir archivos `.wav` para realizar la inferencia.

## Descripción de los Modelos

### model.py — CNN 1D (Audio Crudo)
- Procesa directamente las formas de onda del audio.  
- Aprende patrones temporales mediante capas convolucionales 1D.  
- Requiere remuestreo y normalización de la señal de entrada.  
- Entrena más rápido, pero es más sensible al ruido.

### model2.py — CNN 2D (Características MFCC)
- Utiliza los Coeficientes Cepstrales en la Frecuencia de Mel (MFCC) como características de entrada.  
- Emplea una arquitectura CNN 2D similar a la utilizada en clasificación de imágenes.  
- Ofrece mejor capacidad de generalización y resultados más estables.  
- Este modelo obtuvo el mejor rendimiento general y corresponde a `best_audio_model.pt`.

---

## Pruebas e Inferencia

El script `test.py` permite probar el modelo entrenado subiendo archivos `.wav`.

### Uso
```bash
python test.py --model best_audio_model.pt --audio ruta/al/archivo.wav
