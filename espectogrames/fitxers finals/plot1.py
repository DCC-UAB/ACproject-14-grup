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

def generar_comparacio_plots(resultats_original, resultats_augment, resultats_hog):
    """
    Genera comparació visual d'accuracies dels tres processaments:
    original, augmentat i HOG features.
    """
    models = list(resultats_original.keys())
    accuracies_original = [resultats_original[model].get("accuracy", 0) for model in models]
    accuracies_augment = [resultats_augment[model].get("accuracy", 0) for model in models]
    accuracies_hog = [resultats_hog[model].get("accuracy", 0) for model in models]

    x = np.arange(len(models))
    width = 0.25

    # Crear la figura
    plt.figure(figsize=(14, 8))
    plt.bar(x - width, accuracies_original, width=width, label="Original", alpha=0.8)
    plt.bar(x, accuracies_augment, width=width, label="Amb Augmentació", alpha=0.8)
    plt.bar(x + width, accuracies_hog, width=width, label="HOG Features", alpha=0.8)

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Comparació d'Accuracy: Original vs Augmentat vs HOG Features")
    plt.legend()
    plt.tight_layout()

    # Guardar el plot com a fitxer d'imatge
    plt.savefig("comparacio_accuracy.png", dpi=300)  # Guarda com a imatge PNG
    print("[SUCCESS] El plot s'ha guardat com a 'comparacio_accuracy.png'.")
    plt.show()


if __name__ == "__main__":
    # Fitxers JSON per als tres tipus de processament
    fitxer_original = "C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/fitxers finals/totsmodels_original.json"
    fitxer_augmentat = "C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/fitxers finals/totsmodels_augment.json"
    fitxer_hog = "C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/fitxers finals/totsmodels_hogFeatures.json"

    # Carregar resultats
    print("[INFO] Carregant resultats...")
    resultats_original = carregar_resultats(fitxer_original)
    resultats_augmentat = carregar_resultats(fitxer_augmentat)
    resultats_hog = carregar_resultats(fitxer_hog)

    # Generar el plot comparatiu
    print("[INFO] Generant plots comparatius...")
    generar_comparacio_plots(resultats_original, resultats_augmentat, resultats_hog)
    print("[SUCCESS] Plots generats correctament!")
