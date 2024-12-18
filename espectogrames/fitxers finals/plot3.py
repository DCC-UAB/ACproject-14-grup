import json
import matplotlib.pyplot as plt
import numpy as np

def carregar_resultats(nom_fitxer):
    """Carrega els resultats des del fitxer JSON."""
    with open(nom_fitxer, "r") as fitxer:
        return json.load(fitxer)

def generar_comparacio_models(resultats_sense_gs, resultats_amb_gs):
    """
    Genera comparacions visuals per als 3 millors models amb i sense Grid Search.
    """
    models = ["Logistic Regression", "Support Vector Machine (SVM)", "XGBoost (XGB)"]
    
    # Valors sense Grid Search (només els 3 millors models)
    accuracy_sense_gs = [resultats_sense_gs.get(model, {}).get("accuracy", 0) for model in models]
    temps_sense_gs = [resultats_sense_gs.get(model, {}).get("temps_predict", 0) for model in models]
    
    # Valors amb Grid Search
    accuracy_amb_gs = [resultats_amb_gs[model]["test_accuracy"] for model in models]
    temps_amb_gs = [resultats_amb_gs[model]["cross_val_mean"] for model in models]

    x = np.arange(len(models))
    width = 0.35

    # Comparació d'accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accuracy_sense_gs, width=width, label="Sense Grid Search", alpha=0.8)
    plt.bar(x + width/2, accuracy_amb_gs, width=width, label="Amb Grid Search", alpha=0.8)
    plt.xticks(x, models)
    plt.ylabel("Accuracy")
    plt.title("Comparació d'Accuracy: Sense GS vs Amb GS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparacio_accuracy_gs.png", dpi=300)
    print("[SUCCESS] El plot d'accuracy s'ha guardat com 'comparacio_accuracy_gs.png'.")
    plt.show()

    # Comparació de temps de predicció
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, temps_sense_gs, width=width, label="Sense Grid Search", alpha=0.8)
    plt.bar(x + width/2, temps_amb_gs, width=width, label="Amb Grid Search", alpha=0.8)
    plt.xticks(x, models)
    plt.ylabel("Temps de Predicció (s)")
    plt.title("Comparació de Temps de Predicció: Sense GS vs Amb GS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparacio_temps_gs.png", dpi=300)
    print("[SUCCESS] El plot de temps s'ha guardat com 'comparacio_temps_gs.png'.")
    plt.show()


if __name__ == "__main__":
    # Fitxers JSON
    fitxer_sense_gs = "C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/fitxers finals/totsmodels_hogFeatures.json"
    fitxer_amb_gs = "C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/fitxers finals/bestModels_GS+CV_hog.json"     # Resultats amb Grid Search i Cross Validation
    

    # Carregar resultats
    print("[INFO] Carregant resultats...")
    resultats_sense_gs = carregar_resultats(fitxer_sense_gs)
    resultats_amb_gs = carregar_resultats(fitxer_amb_gs)

    # Generar comparacions
    print("[INFO] Generant comparacions visuals...")
    generar_comparacio_models(resultats_sense_gs, resultats_amb_gs)
    print("[SUCCESS] Plots generats i guardats correctament!")
