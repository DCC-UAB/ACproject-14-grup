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
                .replace("Decision Trees", "Trees")
                .replace("Random Forest", "Forest")
                .replace("Cross Gradient Booster", "Cross GB")
                .replace("(Random Forest)", "RF") for name in names]

# Funció per generar un gràfic comparatiu per a una mètrica específica
def plot_metric_comparison(models, metric, values_first, values_second, title, ylabel):
    """
    Gràfic comparatiu per a una mètrica específica entre dos JSONs.
    
    Args:
    - models (list): Llista de noms dels models.
    - metric (str): Nom de la mètrica.
    - values_first (list): Valors del primer JSON.
    - values_second (list): Valors del segon JSON.
    - title (str): Títol del gràfic.
    - ylabel (str): Etiqueta de l'eix Y.
    """
    x = np.arange(len(models))  # Índexs per als models
    width = 0.35  # Amplada de les barres

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, values_first, width, label="Primer JSON", color="skyblue", edgecolor="black")
    plt.bar(x + width/2, values_second, width, label="Segon JSON", color="orange", edgecolor="black")

    plt.xticks(x, shorten_model_names(models), rotation=20, fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carregar els JSONs
    current_dir = Path(__file__).parent.resolve()
    json_first_path = current_dir / "resultats_primera_prova.json"
    json_second_path = current_dir / "resultats.json"

    json_first = carregar_json(json_first_path)
    json_second = carregar_json(json_second_path)

    # Models i mètriques a comparar
    models = list(json_first["audio"].keys())
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    # Generar un gràfic per cada mètrica
    for metric in metrics:
        values_first = [json_first["audio"][model][metric][0] if json_first["audio"][model][metric] else 0 for model in models]
        values_second = [json_second["audio"][model][metric][0] if json_second["audio"][model][metric] else 0 for model in models]
        
        plot_metric_comparison(models, metric, values_first, values_second, 
                               f"Comparació de {metric} entre dos JSONs", metric.capitalize())
