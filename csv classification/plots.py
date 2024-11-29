import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

train_shapes_3s = [8991, 7992, 6993, 5994, 4995]  # Nombre d'instàncies al conjunt d'entrenament
test_shapes_3s = [999, 1998, 2997, 3996, 4995]    # Nombre d'instàncies al conjunt de test

gaussian_nb_3s = [0.50651, 0.51151, 0.51518, 0.51652, 0.51572]
bernoulli_nb_3s = [0.08609, 0.09109, 0.09009, 0.09484, 0.0967]
multinomial_nb_3s = [0.48248, 0.502, 0.50384, 0.51176, 0.51912]
svm_3s = [0.75676, 0.74825, 0.74775, 0.73524, 0.73233]
knn_3s = [0.84585, 0.81381, 0.80414, 0.78103, 0.76737]
decision_trees_3s = [0.66567, 0.64314, 0.65265, 0.63313, 0.61782]
random_forest_3s = [0.81682, 0.80931, 0.80814, 0.8048, 0.7952]
gradient_boosting_3s = [0.82983, 0.83133, 0.82649, 0.81782, 0.81081]
cross_gradient_boosting_3s = [0.92593, 0.91692, 0.90958, 0.8959, 0.88308]
cross_gb_rf_3s = [0.73173, 0.73874, 0.74041, 0.73924, 0.74955]


train_shapes_30s = [900, 800, 700, 600, 500]  # Nombre d'instàncies al conjunt d'entrenament
test_shapes_30s = [100, 200, 300, 400, 500]   # Nombre d'instàncies al conjunt de test

gaussian_nb_30s = [0.56, 0.565, 0.56333, 0.545, 0.536]
bernoulli_nb_30s = [0.05, 0.07, 0.08, 0.09, 0.092]
multinomial_nb_30s = [0.53, 0.54, 0.51, 0.505, 0.484]
svm_30s = [0.76, 0.73, 0.71333, 0.6975, 0.622]
knn_30s = [0.67, 0.65, 0.61667, 0.6175, 0.552]
decision_trees_30s = [0.63, 0.545, 0.61333, 0.5625, 0.568]
random_forest_30s = [0.85, 0.805, 0.78, 0.765, 0.744]
gradient_boosting_30s = [0.85, 0.815, 0.76, 0.7575, 0.718]
cross_gradient_boosting_30s = [0.86, 0.835, 0.78333, 0.775, 0.744]
cross_gb_rf_30s = [0.79, 0.76, 0.71, 0.7025, 0.64]


# Dades resumides (30s i 3s combinats, ajusta segons el teu cas)


results = {
    "test_size": [0.1, 0.2, 0.3, 0.4, 0.5],
    "3s_mean_accuracy": [0.77891, 0.76246, 0.75868, 0.74438, 0.73819],  # Promedi dels models de les dades 3s
    "30s_mean_accuracy": [0.724, 0.7152, 0.68933, 0.668, 0.625]  # Promedi dels models de les dades 30s
}

# Convertim a DataFrame
df = pd.DataFrame(results)

# Gràfic comparatiu
plt.figure(figsize=(10, 6))
plt.plot(df["test_size"], df["3s_mean_accuracy"], label="3 seconds data", marker="o")
plt.plot(df["test_size"], df["30s_mean_accuracy"], label="30 seconds data", marker="s")

# Configuració del gràfic
plt.title("Accuracy mitjana per diferents test sizes")
plt.xlabel("Test size (% del dataset)")
plt.ylabel("Accuracy mitjana")
plt.xticks(df["test_size"])
plt.legend()
plt.grid(True)

# Mostrar el gràfic
plt.show()
