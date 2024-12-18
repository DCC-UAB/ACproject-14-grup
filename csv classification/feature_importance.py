from pathlib import Path
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data

def definirXY_normalitzar(data):
    X = data.drop(['label', 'filename'], axis=1)  # Treure label i filename
    y = data['label']  # Variable dependent (a predir)
    columnes = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)  # Escalem
    X = pd.DataFrame(np_scaled, columns=columnes)  # Nou dataset sense label ni filename
    return X, y

def divisio_dades(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=111)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape} (test_size={test_size})")  # Mostrem dimensions
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    current_dir = Path(__file__).parent

    # Construir el camí als csv
    cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

    data30s = pd.read_csv(cami_csv_30s)

    data = codificar_label(data30s)  # Modifiquem aquí per treballar amb features_30_sec.csv
    X, y = definirXY_normalitzar(data)

    X_train, X_test, y_train, y_test = divisio_dades(X, y)

    # Entrenament del model
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=1)
    xgb.fit(X_train, y_train)

    # Importància per permutació
    result = permutation_importance(xgb, X_test, y_test, n_repeats=10, random_state=1, scoring="accuracy")

    # Ordenar importàncies
    feature_importances = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Mostrem la importància de les característiques
    print("\n--- Importància de les característiques ---")
    print(feature_importances)

    # Gràfic de barres
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color="skyblue")
    plt.xlabel("Importància", fontsize=14)
    plt.ylabel("Característica", fontsize=14)
    plt.title("Importància de les característiques per permutació", fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()

    # Selecció de les 6 columnes més importants
    top_features = feature_importances['Feature'].head(6).tolist()
    print(f"\nLes 6 columnes més importants són: {top_features}")

    # Crear un nou DataFrame només amb les columnes seleccionades
    data_top_features = data30s[['filename', 'label'] + top_features]  # Incloem filename i label per context

    # Guardar el nou CSV
    nou_cami_csv = current_dir.parent / "datasets" / "Data1" / "features_30_sec_top6.csv"
    data_top_features.to_csv(nou_cami_csv, index=False)
    print(f"\nNou CSV creat: {nou_cami_csv}")
