import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from librosa.feature import rhythm

def extract_features(audio_path, genre, segment = None):
    """
    Extreu les característiques d'un fitxer d'àudio utilitzant Librosa.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Carrega l'àudio
        if segment is not None:
            start_sample = int(segment[0] * sr)
            end_sample = int(segment[1] * sr)
            y = y[start_sample:end_sample]

        features = {
            'genre': genre,  # genere --> serà valor a preddir
            'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rmse': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
            'tempo': float(rhythm.tempo(y=y, sr=sr)),  
            'spectral_contrast': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
            'chroma_cqt': np.mean(librosa.feature.chroma_cqt(y=y, sr=sr)),
        }
        harmonic = librosa.effects.harmonic(y)
        features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=harmonic, sr=sr))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc{i+1}'] = np.mean(mfcc)
        return features
        
    except Exception as e:
        print(f"Error processant {audio_path}: {e}")
        return None

def process_audio_directory(directory, output_csv, segment_duration = 10): # treure features de cada audio (trobarlos)
    """
    Processa tots els fitxers d'àudio en subdirectoris i guarda les característiques en un CSV.
    """
    data = []
    base_path = Path(directory).resolve()  # Converteix el directori en un Path absolut
    for genre_folder in base_path.iterdir():  # recorre els subdirectoris (blues, classical, ...)
        if genre_folder.is_dir():  
            for file in genre_folder.iterdir():  # recorre els fitxers dins del subdirectori
                if file.suffix == '.wav':  # Només processa fitxers .wav
                    try:
                        y, sr = librosa.load(str(file), sr=None)
                        total_duration = librosa.get_duration(y=y, sr=sr)

                        # Dividir l'àudio en segments
                        num_segments = int(total_duration // segment_duration)
                        for i in range(num_segments):
                            start = i * segment_duration
                            end = start + segment_duration
                            features = extract_features(str(file), genre_folder.name, segment=(start, end))
                            if features is not None:
                                features['segment'] = f"{start}-{end}"  # Marca el segment processat
                                data.append(features)
                                
                    except Exception as e:
                        print(f"Error processant {file}: {e}")  # Mostra el fitxer problemàtic però continua


    # Convertir les dades en un DataFrame i guardar-les en un CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Característiques guardades a: {output_csv}")

current_dir = Path(__file__).parent.resolve()  
base_directory = current_dir / '../datasets/Data1/genres_original/'  # Ruta als àudios
output_csv = current_dir / '../audio classification/audio_features_prova2.csv'  # Ruta del CSV de sortida

process_audio_directory(base_directory, output_csv, segment_duration =10)
