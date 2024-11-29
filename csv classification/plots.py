import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_shapes_3s = [8991, 7992, 6993, 5994, 4995]  # Nombre d'instàncies al conjunt d'entrenament
test_shapes_3s = [999, 1998, 2997, 3996, 4995]    # Nombre d'instàncies al conjunt de test
train_shapes_30s = [900, 800, 700, 600, 500]  # Nombre d'instàncies al conjunt d'entrenament
test_shapes_30s = [100, 200, 300, 400, 500]   # Nombre d'instàncies al conjunt de test

# Dades dels resultats de 3 segons
data_3s = {
    "Gaussian NB": [0.50651, 0.51151, 0.51518, 0.51652, 0.51572],
    "Bernoulli NB": [0.08609, 0.09109, 0.09009, 0.09484, 0.0967],
    "Multinomial NB": [0.48248, 0.502, 0.50384, 0.51176, 0.51912],
    "SVM": [0.75676, 0.74825, 0.74775, 0.73524, 0.73233],
    "KNN": [0.84585, 0.81381, 0.80414, 0.78103, 0.76737],
    "Decision Trees": [0.66567, 0.64314, 0.65265, 0.63313, 0.61782],
    "Random Forest": [0.81682, 0.80931, 0.80814, 0.8048, 0.7952],
    "Gradient Boosting": [0.82983, 0.83133, 0.82649, 0.81782, 0.81081],
    "Cross Gradient Boosting": [0.92593, 0.91692, 0.90958, 0.8959, 0.88308],
    "Cross GB RF": [0.73173, 0.73874, 0.74041, 0.73924, 0.74955]
}

# Dades dels resultats de 30 segons
data_30s = {
    "Gaussian NB": [0.56, 0.565, 0.56333, 0.545, 0.536],
    "Bernoulli NB": [0.05, 0.07, 0.08, 0.09, 0.092],
    "Multinomial NB": [0.53, 0.54, 0.51, 0.505, 0.484],
    "SVM": [0.76, 0.73, 0.71333, 0.6975, 0.622],
    "KNN": [0.67, 0.65, 0.61667, 0.6175, 0.552],
    "Decision Trees": [0.63, 0.545, 0.61333, 0.5625, 0.568],
    "Random Forest": [0.85, 0.805, 0.78, 0.765, 0.744],
    "Gradient Boosting": [0.85, 0.815, 0.76, 0.7575, 0.718],
    "Cross Gradient Boosting": [0.86, 0.835, 0.78333, 0.775, 0.744],
    "Cross GB RF": [0.79, 0.76, 0.71, 0.7025, 0.64]
}

# Convertim les dades en DataFrames
df_3s = pd.DataFrame(data_3s)
df_30s = pd.DataFrame(data_30s)

# Assignem índexs per indicar la mida del test size
# df_3s.index = [0.1, 0.2, 0.3, 0.4, 0.5]
# df_30s.index = [0.1, 0.2, 0.3, 0.4, 0.5]

# Concatenem els DataFrames
dataframe_total = (df_3s+df_30s) / 2 # mitjana de cada model sense tenir ecompte diferencia entre 3s i 30s

# Calculem la mitjana per test size
# column_means = df_combined.mean(axis=1)


# print("Mitjanes per test size (3s i 30s):")
# print(column_means)

# # Opcional: si vols veure la taula completa
# print("\nDataFrame combinat:")
# print(df_combined)

def plot_accuracy(dataframe, titol = "Default"):

    plt.figure(figsize=(12, 6))
    bars = dataframe.mean(axis=0).plot(kind="bar", color="skyblue", alpha=0.9, edgecolor='black')
    plt.title(f"Accuracy per model {titol }", fontsize=16)
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
    
    Args:
    - df_3s (DataFrame): Dades del dataset de 3 segons.
    - df_30s (DataFrame): Dades del dataset de 30 segons.
    - titol (str): Títol del gràfic.
    """
    # Calcular la mitjana per model
    mean_3s = df_3s.mean(axis=0)
    mean_30s = df_30s.mean(axis=0)

    # Combinar les dades en un únic DataFrame
    comparison_df = pd.DataFrame({
        "3 segons": mean_3s,
        "30 segons": mean_30s
    })

    # Crear el gràfic
    comparison_df.plot(kind="bar", figsize=(12, 6), color=["skyblue", "orange"], edgecolor='black', alpha=0.9)
    
    plt.title(f"Accuracy per model: {titol}", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Dataset", fontsize=12)
    plt.show()

# plot_accuracy(dataframe_total, '')
# plot_accuracy_comparison(df_3s, df_30s, titol="Dades de 3 segons vs 30 segons")



