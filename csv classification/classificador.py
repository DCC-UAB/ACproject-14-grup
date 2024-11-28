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
cami_csv_3s = os.path.join(current_dir, "..", "datasets", "Data1", "features_3_sec")
cami_csv_30s = os.path.join(current_dir, "..", "datasets", "Data1", "features_3_sec")

data = pd.read_csv(cami_csv_3s)
data = data.iloc[0:, 1:] 
data.head()