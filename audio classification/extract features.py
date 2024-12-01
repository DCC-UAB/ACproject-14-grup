import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

def extract_features(audio_path, genre):
    """
    Extreu les característiques d'un fitxer d'àudio utilitzant Librosa.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Carrega l'àudio
        features = {
            'genre': genre,  # Afegeix el gènere com a columna
            'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rmse': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
            'tempo': float(librosa.beat.tempo(y=y, sr=sr)),  # Ritme (tempo)
        }
        # Extreure MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc{i+1}'] = np.mean(mfcc)
        return features
    except Exception as e:
        print(f"Error processant {audio_path}: {e}")
        return None

def process_audio_directory(directory, output_csv):
    """
    Processa tots els fitxers d'àudio en subdirectoris i guarda les característiques en un CSV.
    """
    data = []
    base_path = Path(directory).resolve()  # Converteix el directori en un Path absolut
    for genre_folder in base_path.iterdir():  # Recorre els subdirectoris (blues, classical, ...)
        if genre_folder.is_dir():  # Assegura que és un subdirectori
            for file in genre_folder.iterdir():  # Recorre els fitxers dins del subdirectori
                if file.suffix == '.wav':  # Només processa fitxers .wav
                    features = extract_features(str(file), genre_folder.name)
                    if features is not None:
                        data.append(features)

    # Convertir les dades en un DataFrame i guardar-les en un CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Característiques guardades en: {output_csv}")

# Exemple d'ús
current_dir = Path(__file__).parent.resolve()  # Directori actual del script
base_directory = current_dir / '../datasets/Data1/genres_original/'  # Ruta als àudios
output_csv = current_dir / '../datasets/audio_features.csv'  # Ruta del CSV de sortida

process_audio_directory(base_directory, output_csv)
