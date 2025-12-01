# Clasificación de Audio de Gatos y Perros

Este proyecto implementa y compara dos enfoques de aprendizaje profundo para la clasificación de clips de audio de gatos y perros.  
Los modelos fueron desarrollados con PyTorch y probados mediante dos scripts que permiten subir archivos `.wav` para realizar la inferencia.

## Reporte de resultados
Con el fin de poder tener un completo entendimiento de la metodología seguida para edesarrollo de ambos modelos, se recomienda leer el archivo: 

`Reporte.pdf`

Contenido en el presente repositorio. 

## Descripción de los Modelos

### OneD_CNN_model.py — CNN 1D (Audio Crudo)
- Procesa directamente las formas de onda del audio.  
- Aprende patrones temporales mediante capas convolucionales 1D.  
- Requiere remuestreo y normalización de la señal de entrada.  
- Entrena más rápido, pero es más sensible al ruido.

### MFCC_model.py — CNN 2D (Características MFCC)
- Utiliza los Coeficientes Cepstrales en la Frecuencia de Mel (MFCC) como características de entrada.  
- Emplea una arquitectura CNN 2D similar a la utilizada en clasificación de imágenes.  
- Propuesta de mejora para mejora para mayor abstracción de caracteristicas de audio. 

## Instalación de Dependencias

Con el fin de poder ejecutar ambas implementaciones de modelo desarrolladas, se rcomienda instalar las 
dependencias contenidas en el archivo `requirements.txt`contenido el presente repositorio. El comando sugerido para
realizar la isntalación se muestra a continuación:

```bash
pip install -r requirements.txt
```

## Instrucciones para Realizar Pruebas e Inferencias

Con el fin de poder realizar puebas e inferencias en torno a los modelos presentes en este repositorio, se recomienda tener los siguienres archivos y carpetas en un mismo directorio: 

- `OneD_CNN_model.py`
- `MFCC_CNN_model.py`
- `OneD_CNN_test.py`
- `MFCC_CNN_test.py`
- `cats_dogs`

Ambas implementaciones atomaticamente detectan los directorios de entrenamiento y prueba contenidos en la carpeta `cats_dogs` por ende el énfasis en que se realice el entrnamiento de ambos modelo dentro de la misma carpetat. 

Una vez el entrenamiento de ambos modelos se haya realizado, dos archivos nombrados `best_audio_model.pt` serán generados rn rl mismo directorio. dentro de los scripts `MFCC_CNN_test.py` y `MFCC_CNN_test.py` ya se cuenta con la ruta prestablecida para la lectura de estos archivos generados. 

Una vez se hayan generados estos archivos, el comando a ejecutar en terminal es el siguiente para cada caso de modelo: 

### Uso
```bash
python OneD_CNN_model.py --model best_audio_model.pt --audio ruta/al/archivo.wav
```
```bash
python MFCC_CNN_model.py --model best_audio_model.pt --audio ruta/al/archivo.wav
```

La ruta del archivo `.wav`, se deja a consideración de cada usuario, ya que le sistema es capaz de leer cualquier archivo de esta extensión para pobar cualquiera de ambos modelos. 