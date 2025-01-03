import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Carregar resultats des del JSON

current_dir = Path(__file__).parent.resolve()

# Carregar el JSON principal (dataset sencer)
nom_fitxer_json = current_dir / "resultats_all_models.json"

with open(nom_fitxer_json, "r") as fitxer:
    resultats = json.load(fitxer)

# Convertir resultats en DataFrames
df_3s = pd.DataFrame(resultats["3 seconds"]).T  # Transposar per accedir als models com a files
df_30s = pd.DataFrame(resultats["30 seconds"]).T

# Funció per escurçar els noms dels models
def shorten_model_names(names):
    """
    Escurça els noms dels models eliminant paraules llargues.
    """
    return [name.replace("Naive Bayes", "NB")
                .replace("Support Vector Machine", "SVM")
                .replace("K-Nearest Neighbors", "KNN")
                .replace("Decision Trees", "Decision Trees")
                .replace("Random Forest", "Random Forest")
                .replace("Cross Gradient Booster", "Cross GB")
                .replace("(Random Forest)", "RF") for name in names]

# Funció per crear un gràfic comparatiu de barres per una mètrica
def plot_comparative_bar(df_3s, df_30s, metric, title, ylabel):
    """
    Gràfic comparatiu de barres per una mètrica específica.
    
    Args:
    - df_3s (DataFrame): Resultats de 3 segons.
    - df_30s (DataFrame): Resultats de 30 segons.
    - metric (str): Mètrica a visualitzar.
    - title (str): Títol del gràfic.
    - ylabel (str): Etiqueta de l'eix Y.
    """
    comparison_df = pd.DataFrame({
        "3 segons": df_3s[metric],
        "30 segons": df_30s[metric]
    })

    comparison_df.index = shorten_model_names(comparison_df.index)  # Escurçar els noms dels models

    comparison_df.plot(kind="bar", figsize=(10, 5), color=["skyblue", "orange"], edgecolor='black', alpha=0.9)
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=20, fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Dataset", fontsize=10)
    plt.tight_layout()
    plt.show()

# Generar un gràfic per cada mètrica
metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
for metric in metrics:
    plot_comparative_bar(df_3s, df_30s, metric, f"Comparació de {metric} entre models", metric.capitalize())
