import json
import matplotlib.pyplot as plt
import numpy as np
import os

def carregar_resultats(nom_fitxer):
    """
    Carrega els resultats des del fitxer JSON.
    """
    with open(nom_fitxer, "r") as fitxer:
        resultats = json.load(fitxer)
    return resultats

def guardar_plot(nom_directori, nom_fitxer):
    """
    Guarda el plot actual al directori especificat.
    """
    if not os.path.exists(nom_directori):
        os.makedirs(nom_directori)  # Crear el directori si no existeix
    plt.savefig(os.path.join(nom_directori, nom_fitxer), bbox_inches='tight')
    print(f"[INFO] Guardant plot a {os.path.join(nom_directori, nom_fitxer)}")

def generar_plots(resultats):
    """
    Genera comparacions visuals dels resultats i guarda els plots.
    """
    nom_directori_plots = "output_plots"
    models = list(resultats.keys())
    accuracies = [resultats[model].get("test_accuracy", 0) for model in models]
    cross_val_means = [resultats[model].get("cross_val_mean", 0) for model in models]
    best_scores = [resultats[model].get("best_accuracy", 0) for model in models]

    x = np.arange(len(models))

    # Comparació de test accuracy i cross-validation accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, accuracies, width=0.4, label="Test Accuracy", alpha=0.8)
    plt.bar(x + 0.2, cross_val_means, width=0.4, label="Cross-Validation Mean", alpha=0.8)
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Comparació d'Accuracy: Test vs Cross-Validation")
    plt.legend()
    plt.tight_layout()
    guardar_plot(nom_directori_plots, "comparacio_test_vs_cross_validation.png")
    plt.show()

    # Comparació de millors scores trobats per Grid Search
    plt.figure(figsize=(12, 6))
    plt.bar(models, best_scores, color="green", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Best Accuracy")
    plt.title("Millors Scores trobats per Grid Search")
    plt.tight_layout()
    guardar_plot(nom_directori_plots, "best_scores_grid_search.png")
    plt.show()

if __name__ == "__main__":
    nom_fitxer = "bestModels_GS+CV_hog.json" 
    print(f"[INFO] Carregant resultats de {nom_fitxer}...")
    resultats = carregar_resultats("C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/fitxers finals/bestModels_GS+CV_hog.json"
)

    print("[INFO] Generant plots de comparació i guardant-los...")
    generar_plots(resultats)
