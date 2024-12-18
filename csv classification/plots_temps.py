import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Funció per carregar un fitxer JSON
def carregar_json(nom_fitxer):
    with open(nom_fitxer, "r") as f:
        return json.load(f)

# Funció per generar un gràfic combinat d'entrenament i predicció
def plot_combined_times(train_before, train_after, pred_before, pred_after, title, ylabel, labels):
    """
    Gràfic combinat per temps d'entrenament i de predicció.
    
    Args:
    - train_before (list): Temps d'entrenament abans.
    - train_after (list): Temps d'entrenament després.
    - pred_before (list): Temps de predicció abans.
    - pred_after (list): Temps de predicció després.
    - title (str): Títol del gràfic.
    - ylabel (str): Etiqueta de l'eix Y.
    - labels (list): Etiquetes per cada model.
    """
    x = np.arange(len(labels))  # Índexs per als models
    width = 0.2  # Amplada de les barres

    plt.figure(figsize=(12, 6))

    # Barres d'entrenament
    plt.bar(x - 1.5*width, train_before, width, label="Train (DS original)", color="skyblue", edgecolor="black")
    plt.bar(x - 0.5*width, train_after, width, label="Train (Dades netes)", color="orange", edgecolor="black")

    # Barres de predicció
    plt.bar(x + 0.5*width, pred_before, width, label="Pred (DS original)", color="lightgreen", edgecolor="black")
    plt.bar(x + 1.5*width, pred_after, width, label="Pred (Dades netes)", color="lightcoral", edgecolor="black")

    plt.xticks(x, labels, rotation=20, fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    current_dir = Path(__file__).parent.resolve()

    # Carregar el JSON principal (dataset sencer)
    fitxer_og = current_dir / "temps_models_fitxer_OG.json"
    temps_og = carregar_json(fitxer_og)

    # Carregar els JSONs nets (un per cada model)
    fitxers_nets = {
        "Random Forest": current_dir / "temps_RF_fitxer_net_Ramon.json",
        "Gradient Boosting": current_dir / "temps_GB_fitxer_net_Ramon.json",
        "Cross Gradient Booster": current_dir / "temps_XGB_fitxer_net_Ramon.json"
    }

    temps_net = {model: carregar_json(fitxer) for model, fitxer in fitxers_nets.items()}

    # Models a comparar
    models = ["Random Forest", "Gradient Boosting", "Cross Gradient Booster"]

    # Extreure dades per als tres models
    train_times_og = [temps_og["3 seconds"][model]["train_time"] for model in models]
    pred_times_og = [temps_og["3 seconds"][model]["pred_time"] for model in models]

    train_times_net = [temps_net[model]["3 seconds"][model]["train_time"] for model in models]
    pred_times_net = [temps_net[model]["3 seconds"][model]["pred_time"] for model in models]

    accuracy_og = [temps_og["3 seconds"][model]["accuracy"] for model in models]
    accuracy_net = [temps_net[model]["3 seconds"][model]["accuracy"] for model in models]

    # Generar el gràfic combinat per temps d'entrenament i predicció
    plot_combined_times(train_times_og, train_times_net, 
                        pred_times_og, pred_times_net, 
                        "Comparació del temps d'entrenament i predicció", 
                        "Temps (segons)", models)

    # Generar el gràfic d'accuracy
    plot_combined_times(accuracy_og, accuracy_net, 
                        [0]*len(models), [0]*len(models), 
                        "Comparació de l'accuracy", "Accuracy", models)
