import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile as wav
import pandas as pd
import os
import numpy as np
import seaborn as sns
from PIL import Image

# Obtenir el directori actual del fitxer (startingPoints)
current_dir = os.path.dirname(__file__)

# Construir el camí relatiu a "datasets"
datasets_dir = os.path.join(current_dir, "..", "datasets")

# Afegir el camí al fitxer específic
cami_audio = os.path.join(datasets_dir, "Data1", "genres_original", "blues", "blues.00000.wav")
cami_espectograma = os.path.join(datasets_dir, "Data1", "images_original", "blues", "blues00000.png")

#funcio que dibuixa les ones 
def plot_sound(path):
    plt.figure(figsize=(14, 5))
    x, sr = librosa.load(path)
    print("length {}, sample-rate {}".format(x.shape, sr)) # sr: number of samples of audio carried per second, measured in Hz or kHz
    librosa.display.waveshow(x, sr=sr, alpha=0.8) 
    plt.title("Ona de l'àudio")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitud")
    plt.show()  # Mostrar el gràfic
    return x


blues_audio = plot_sound(cami_audio)
ipd.Audio(cami_audio)

def plot_spectograma(path):
    # Obrir el fitxer PNG
    image = Image.open(path)
    # Mostrar la imatge
    image.show()

espectograma_exemple=plot_spectograma(cami_espectograma)


