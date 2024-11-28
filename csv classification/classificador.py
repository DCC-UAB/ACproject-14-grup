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
import pandas as pd

# Obtenir el directori actual del fitxer
current_dir = Path(__file__).parent

# Construir el camí als csv
cami_csv_3s = current_dir.parent / "datasets" / "Data1" / "features_3_sec.csv"
cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

data = pd.read_csv(cami_csv_3s)
data = data.iloc[0:, 1:] 
print(data.head())