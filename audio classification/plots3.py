import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Carregar JSONs
def carregar_json(nom_fitxer):
    with open(nom_fitxer, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Carregar JSONs
    current_dir = Path(__file__).parent.resolve()
    json_optimized_path = current_dir / "resultats_support_vector_machine_best_hyperparametresP1.json"
    json_svm_path = current_dir / "resultats.json"

    optimized_svm = carregar_json(json_optimized_path)
    svm_models = carregar_json(json_svm_path)

    # Extreure dades
    accuracy_mean = 0.9495  # Accuracy mitjana del SVM manual
    optimized_svm_accuracy = optimized_svm["audio"]["Optimized SVMClassifier"]["accuracy"]
    svm_json_accuracy = svm_models["audio"]["Support Vector Machine"]["accuracy"][0]

    # Crear llistes per al gràfic
    models = ["SVM inici", "Grid Search SVM", "Cross Validation SVM"]
    accuracies = [
        svm_json_accuracy,
        optimized_svm_accuracy,
        accuracy_mean
    ]

    # Gràfic
    def plot_accuracy(models, accuracies):
        """
        Gràfic de barres per a l'accuracy de SVM en diferents casos.
        
        Args:
        - models (list): Llista de noms dels models.
        - accuracies (list): Valors d'accuracy per cada model.
        """
        x = np.arange(len(models))  # Índexs per als models
        width = 0.5  # Amplada de les barres

        plt.figure(figsize=(10, 6))
        plt.bar(x, accuracies, width, color="skyblue", edgecolor="black")

        plt.xticks(x, models, rotation=20, fontsize=12)
        plt.ylabel("Accuracy", fontsize=14)
        plt.ylim(0, 1.1)  # Limitar l'eix Y entre 0 i 1.1
        plt.title("Comparació de l'Accuracy de SVM en diferents casos", fontsize=16)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Generar el gràfic
    plot_accuracy(models, accuracies)
