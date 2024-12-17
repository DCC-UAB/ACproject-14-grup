import json
import matplotlib.pyplot as plt
import numpy as np

# Funcions per carregar resultats i generar comparacions visuals
def carregar_resultats(nom_fitxer):
    """
    Carrega els resultats des del fitxer JSON.
    """
    with open(nom_fitxer, "r") as fitxer:
        resultats = json.load(fitxer)
    return resultats

def generar_comparacio_plots(resultats_original, resultats_augmentat, resultats_hog):
    """
    Genera comparacions visuals entre els tres tipus de processament de dades.
    """
    models = list(resultats_original.keys())

    # Extreure l'accuracy de cada processament
    accuracies_original = [resultats_original[model].get("accuracy", 0) for model in models]
    accuracies_augmentat = [resultats_augmentat[model].get("accuracy", 0) for model in models]
    accuracies_hog = [resultats_hog[model].get("accuracy", 0) for model in models]

    x = np.arange(len(models))

    # Generar el gràfic comparatiu
    plt.figure(figsize=(14, 7))
    plt.bar(x - 0.25, accuracies_original, width=0.25, label="Original")
    plt.bar(x, accuracies_augmentat, width=0.25, label="Augmentat")
    plt.bar(x + 0.25, accuracies_hog, width=0.25, label="HOG Features")

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Comparació d'Accuracy: Original vs Augmentat vs HOG Features")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Fitxers JSON per als tres tipus de processament
    fitxer_original = "bestModels_original.json"
    fitxer_augmentat = "bestModels_augment.json"
    fitxer_hog = "bestModels_GS+CV_hog.json"

    # Carregar resultats
    print("[INFO] Carregant resultats...")
    resultats_original = carregar_resultats(fitxer_original)
    resultats_augmentat = carregar_resultats(fitxer_augmentat)
    resultats_hog = carregar_resultats(fitxer_hog)

    # Generar el plot comparatiu
    print("[INFO] Generant plots comparatius...")
    generar_comparacio_plots(resultats_original, resultats_augmentat, resultats_hog)
    print("[SUCCESS] Plots generats correctament!")
