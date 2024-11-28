import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import librosa
import librosa.display
import IPython.display as ipd
import warnings
import os

from pathlib import Path

# Obtenir el directori actual del fitxer
current_dir = Path(__file__).parent

# Construir el camí als csv
cami_csv_3s = current_dir.parent / "datasets" / "Data1" / "features_3_sec.csv"
cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

data30s = pd.read_csv(cami_csv_30s)
data = data30s.iloc[0:, 1:] 

print(data['label'].value_counts())

def elimina_outliers_iqr(df, columna):
    # Calculem quartils
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1

    # Definim limits 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrem dades 
    df_sense_outliers = df[(df[columna] >= lower_bound) & (df[columna] <= upper_bound)]
    return df_sense_outliers

# Exemple:
""" data_filtrat = elimina_outliers_iqr(data, 'tempo') #ens basem amb el tempooo per treure els outliers ??

current_dir = Path(__file__).parent  # Aquest codi només funciona en scripts locals

output_file = current_dir / "dades_sense_outliers_tempo_30s.csv"
data_filtrat.to_csv(output_file, index=False) """

dataneta30s = pd.read_csv("csv classification\dades_sense_outliers_tempo_30s.csv")
dataneta = dataneta30s.iloc[0:, 1:] 

print(dataneta['label'].value_counts())