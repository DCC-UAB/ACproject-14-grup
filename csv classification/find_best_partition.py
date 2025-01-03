import json
import pandas as pd
import matplotlib.pyplot as plt



def ajustar_longitud(valors, longitud_objetiu):
    """
    Ajusta la longitud d'una llista a la longitud objectiu, omplint amb None si cal.
    """
    return valors + [None] * (longitud_objetiu - len(valors))



def plot_generalization_analysis(df_accuracy_gap, df_f1_gap, test_sizes):
    """
    Analitza la generalització comparant les particions en termes de train-test gap.
    
    Args:
    - df_accuracy_gap (DataFrame): Diferència entre train i test per Accuracy.
    - df_f1_gap (DataFrame): Diferència entre train i test per F1-Score.
    - test_sizes (list): Proporcions de test size utilitzades.
    """
    generalization_means = pd.DataFrame({
        "Accuracy Gap": df_accuracy_gap.mean(axis=1),
        "F1-Score Gap": df_f1_gap.mean(axis=1)
    })

    generalization_means.index=[f"{int(t * 100)}%" for t in test_sizes]



    # Verificar les dades
    print("\n--- Generalization Gap per partició ---")
    print(generalization_means)

    # Crear el gràfic
    generalization_means.plot(kind="bar", figsize=(12, 6), color=["skyblue", "orange"], alpha=0.9, edgecolor='black')
    plt.title("Generalització: Train-Test Gap per partició (30 segons)", fontsize=16)
    plt.ylabel("Valor del Gap (Train - Test)", fontsize=14)
    plt.xlabel("Test size", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Mètrica", fontsize=12, loc="best")
    plt.show()

def find_best_partition(df_accuracy, df_f1, df_roc_auc, df_accuracy_gap, df_f1_gap, test_sizes):
    """
    Determina la millor partició basant-se en diverses mètriques, donant més pes al Generalization Gap.
    """
    # Normalitzar les mètriques (entre 0 i 1)
    def normalitza(df):
        return (df - df.min()) / (df.max() - df.min())

    norm_accuracy = normalitza(df_accuracy.mean(axis=1))
    norm_f1 = normalitza(df_f1.mean(axis=1))
    norm_roc_auc = normalitza(df_roc_auc.mean(axis=1))
    norm_accuracy_gap = normalitza(-df_accuracy_gap.mean(axis=1))  # Penalització
    norm_f1_gap = normalitza(-df_f1_gap.mean(axis=1))  # Penalització

    # Escalatge exponencial per penalitzacions
    penalitzacio_factor = 4  # Augmentem encara més el pes de les penalitzacions
    norm_accuracy_gap = norm_accuracy_gap ** penalitzacio_factor
    norm_f1_gap = norm_f1_gap ** penalitzacio_factor

    # Reduïm el pes de ROC-AUC
    roc_auc_factor = 0.5
    norm_roc_auc *= roc_auc_factor

    # Calcular puntuació per partició
    partition_scores = pd.DataFrame({
        "Accuracy": norm_accuracy,
        "F1-Score": norm_f1,
        "ROC-AUC": norm_roc_auc,
        "Accuracy Gap (penalització)": norm_accuracy_gap,
        "F1 Gap (penalització)": norm_f1_gap
    })

    partition_scores.index=[f"{int(t * 100)}%" for t in test_sizes]

    # Sumar puntuacions per cada partició
    partition_scores["Total Score"] = (
        partition_scores["Accuracy"] +
        partition_scores["F1-Score"] +
        partition_scores["ROC-AUC"] -
        partition_scores["Accuracy Gap (penalització)"] -
        partition_scores["F1 Gap (penalització)"]
    )

    # Verificar les puntuacions
    print("\n--- Puntuacions totals per partició (ajustades) ---")
    print(partition_scores)

    # Retornar la millor partició
    best_partition = partition_scores["Total Score"].idxmax()
    print(f"\nLa millor partició és: {best_partition}")
    return best_partition


if __name__ == "__main__":

    # Carregar resultats des del fitxer JSON
    nom_fitxer_json = "resultats.json"

    with open(nom_fitxer_json, "r") as fitxer:
        resultats = json.load(fitxer)

    # Convertir els resultats del dataset de 30 segons en un DataFrame
    data_30s = resultats["30 seconds"]

    # Longitud objectiu basada en el nombre de test sizes
    longitud_objetiu = len(next(iter(data_30s.values()))["accuracy"])

    # Crear DataFrames per a les mètriques
    df_30s_accuracy = pd.DataFrame({model: data["accuracy"] for model, data in data_30s.items()})
    df_30s_f1 = pd.DataFrame({model: data["f1_score"] for model, data in data_30s.items()})
    df_30s_precision = pd.DataFrame({model: data["precision"] for model, data in data_30s.items()})
    df_30s_recall = pd.DataFrame({model: data["recall"] for model, data in data_30s.items()})

    # Gestionar `roc_auc` amb models com SVM
    df_30s_roc_auc = pd.DataFrame({
        model: ajustar_longitud(data.get("roc_auc", [None]), longitud_objetiu)
        for model, data in data_30s.items()
    })

    df_accuracy_gap = pd.DataFrame({model: data["accuracy_gap"] for model, data in data_30s.items()})
    df_f1_gap = pd.DataFrame({model: data["f1_gap"] for model, data in data_30s.items()})

    # Test sizes utilitzats
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Visualitzar el generalization gap
    plot_generalization_analysis(df_accuracy_gap, df_f1_gap, test_sizes)

    # Determinar la millor partició
    find_best_partition(df_30s_accuracy, df_30s_f1, df_30s_roc_auc, df_accuracy_gap, df_f1_gap, test_sizes)
