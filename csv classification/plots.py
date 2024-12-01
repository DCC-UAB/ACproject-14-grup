import json
import pandas as pd
import matplotlib.pyplot as plt

# Carregar resultats des del fitxer JSON
nom_fitxer_json = "resultats.json"

with open(nom_fitxer_json, "r") as fitxer:
    resultats = json.load(fitxer)

# Convertir els resultats en DataFrames (només la accuracy)
df_3s = pd.DataFrame({model: data["accuracy"] for model, data in resultats["3 seconds"].items()})
df_30s = pd.DataFrame({model: data["accuracy"] for model, data in resultats["30 seconds"].items()})

# Definir funcions de gràfics
def plot_accuracy(dataframe, titol="Default"):
    """
    Gràfic de barres per visualitzar l'accuracy mitjana per model.
    """
    plt.figure(figsize=(12, 6))
    bars = dataframe.mean(axis=0).plot(kind="bar", color="skyblue", alpha=0.9, edgecolor='black')
    plt.title(f"Accuracy per model {titol}", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Afegim valors a les barres
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, 
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    plt.show()


def plot_accuracy_comparison(df_3s, df_30s, titol="Comparació"):
    """
    Crea un gràfic de barres comparatiu per als dos datasets.
    """
    mean_3s = df_3s.mean(axis=0)
    mean_30s = df_30s.mean(axis=0)

    comparison_df = pd.DataFrame({
        "3 segons": mean_3s,
        "30 segons": mean_30s
    })

    comparison_df.plot(kind="bar", figsize=(12, 6), color=["skyblue", "orange"], edgecolor='black', alpha=0.9)
    
    plt.title(f"Accuracy per model: {titol}", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Dataset", fontsize=12)
    plt.show()


def plot_test_size_impact(df_3s, df_30s, test_sizes, titol="Impacte del test size"):
    """
    Crea un gràfic de línies per comparar l'impacte del test size en els dos datasets.
    
    Args:
    - df_3s (DataFrame): Dades del dataset de 3 segons.
    - df_30s (DataFrame): Dades del dataset de 30 segons.
    - test_sizes (list): Proporcions de test size utilitzades.
    - titol (str): Títol del gràfic.
    """
    # Calcular les mitjanes per test size
    mean_3s = df_3s.mean(axis=1)
    mean_30s = df_30s.mean(axis=1)

    # Crear el gràfic
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, mean_3s, marker='o', label="3 segons", color="skyblue")
    plt.plot(test_sizes, mean_30s, marker='o', label="30 segons", color="orange")
    plt.title(f"{titol}", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Test size", fontsize=14)
    plt.xticks(test_sizes, [f"{int(t * 100)}%" for t in test_sizes], fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Dataset", fontsize=12)
    plt.show()


# Suposem que aquests són els test_sizes utilitzats en el model
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

# Generar els gràfics
plot_accuracy(df_3s, titol="Dades de 3 segons")
plot_accuracy(df_30s, titol="Dades de 30 segons")
plot_accuracy_comparison(df_3s, df_30s, titol="Dades de 3 segons vs 30 segons")
plot_test_size_impact(df_3s, df_30s, test_sizes, titol="Impacte del test size en accuracy")
