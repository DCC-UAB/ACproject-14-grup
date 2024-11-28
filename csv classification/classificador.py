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

current_dir = os.path.dirname(__file__)

# Construir el camí relatiu a "datasets"
datasets_dir = os.path.join(current_dir, "..", "datasets")
print(datasets_dir)

# Afegir el camí al fitxer específic
cami_csv_3s = os.path.join(datasets_dir, "features_3_sec")
cami_csv_30s = os.path.join(datasets_dir, "features_30_sec")

data = pd.read_csv(cami_csv_3s)
data = data.iloc[0:, 1:] 
data.head()