import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
