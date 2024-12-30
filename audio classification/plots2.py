import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Funció per carregar un fitxer JSON
def carregar_json(nom_fitxer):
    with open(nom_fitxer, "r") as f:
        return json.load(f)

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

# Funció per generar un gràfic per a una mètrica específica
def plot_metric(models, values, metric, title, ylabel):
    """
    Gràfic de barres per a una mètrica específica.
    
    Args:
    - models (list): Llista de noms dels models.
    - values (list): Valors de la mètrica per cada model.
    - metric (str): Nom de la mètrica.
    - title (str): Títol del gràfic.
    - ylabel (str): Etiqueta de l'eix Y.
    """
    x = np.arange(len(models))  # Índexs per als models
    width = 0.5  # Amplada de les barres

    plt.figure(figsize=(12, 6))
    plt.bar(x, values, width, label=metric.capitalize(), color="skyblue", edgecolor="black")

    plt.xticks(x, shorten_model_names(models), rotation=20, fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carregar el JSON
    current_dir = Path(__file__).parent.resolve()
    json_path = current_dir / "resultats.json"

    data = carregar_json(json_path)

    # Models i mètriques a processar
    models = list(data["audio"].keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    # Generar un gràfic per cada mètrica
    for metric in metrics:
        values = [data["audio"][model][metric][0] for model in models]
        plot_metric(models, values, metric, f"{metric.capitalize()} per model", metric.capitalize())
