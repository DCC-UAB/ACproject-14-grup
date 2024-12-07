import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display as ipd
from scipy.io import wavfile as wav
import time
import cv2
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

def augment_image(image):
    flipped = np.fliplr(image)
    noisy = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)  
    brighter = np.clip(image * 1.2, 0, 1)
    darker = np.clip(image * 0.8, 0, 1)
    return [image, flipped, noisy, brighter, darker]

def divisio_en_vectors_augmentat(data, labels, img_size=(128,128), block_size=(4,4)):
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)/ 255.0
                    augmented_images = augment_image(img_resized)

                    for aug_img in augmented_images:
                        features = []

                        h, w = aug_img.shape
                        block_h, block_w = block_size
                        for i in range(0, h, block_h):
                            for j in range(0, w, block_w):
                                block = aug_img[i:i + block_h, j:j + block_w]
                                features.append(np.mean(block))
                                features.append(np.std(block))
                                features.append(np.max(block))
                                features.append(np.min(block))

                        data.append(features)
                        labels.append(genre)



def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data, label_encoder

def definirXY_normalitzar(data):
    X = data.drop(['label'],axis=1)
    y = data['label'] #variable independent (a predir)
    columnes = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X) #escalem 
    X = pd.DataFrame(np_scaled, columns=columnes)#nou dataset sense label i filename
    return X, y

def model_assess(model, X_train, X_test, y_train, y_test, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n') #calcular accuracy

def model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats):
    """
    Avaluar un model amb diverses mètriques i guardar els resultats en un diccionari.
    """
    if title not in resultats:
        resultats[title] = {}

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Calcular mètriques per al test
    accuracy_test = round(accuracy_score(y_test, preds), 5)
    precision_test = round(precision_score(y_test, preds, average="weighted", zero_division=0), 5)
    recall_test = round(recall_score(y_test, preds, average="weighted", zero_division=0), 5)
    f1_test = round(f1_score(y_test, preds, average="weighted", zero_division=0), 5)
    
    if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
        roc_auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 5)
    else:
        roc_auc_test = None
    
    # Calcular mètriques per al train
    preds_train = model.predict(X_train)
    accuracy_train = round(accuracy_score(y_train, preds_train), 5)
    f1_train = round(f1_score(y_train, preds_train, average="weighted", zero_division=0), 5)

    # Calcular generalization gap
    accuracy_gap = round(accuracy_train - accuracy_test, 5)
    f1_gap = round(f1_train - f1_test, 5)


    # Afegir mètriques al diccionari
    resultats[title]["accuracy"]=accuracy_test
    resultats[title]["precision"]=precision_test
    resultats[title]["recall"]=recall_test
    resultats[title]["f1_score"]=f1_test
    if roc_auc_test is not None:
        resultats[title]["roc_auc"]=roc_auc_test
    resultats[title]["accuracy_gap"]=accuracy_gap
    resultats[title]["f1_gap"]=f1_gap


def guardar_resultats_a_json(resultats, nom_fitxer="resultats3.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    current_dir = Path(__file__).parent.resolve()  # Directori actual del script
    fitxer_json = current_dir / nom_fitxer  # Camí complet al fitxer JSON

    with open(nom_fitxer, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {fitxer_json}")


if __name__ == "__main__":
    base_dir = "ACproject-14-grup/datasets/Data1/images_original"
    resultats = {}    

    # Llista de models a avaluar:
    models = [
        (GaussianNB(), "Gaussian Naive Bayes"),
        (BernoulliNB(), "Bernoulli Naive Bayes"),
        (MultinomialNB(), "Multinomial Naive Bayes"),
        #(SVC(decision_function_shape="ovo"), "Support Vector Machine"),
        (KNeighborsClassifier(n_neighbors=19), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(), "Decision Trees"),
        (RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0), "Random Forest"),
        #º(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0), "Gradient Boosting"),
        #(XGBClassifier(n_estimators=1000, learning_rate=0.05), "Cross Gradient Booster"),
        (XGBRFClassifier(objective= 'multi:softmax'),"Cross Gradient Booster (Random Forest)" )

    ]
    
    data, labels = [], []

    divisio_en_vectors_augmentat(data, labels, img_size=(128,128), block_size=(4,4))
    
    data = pd.DataFrame(data)
    data["label"] = labels

    from collections import Counter
    print("Distribució de classes inicial:")
    print(Counter(labels))

    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    # validació distribució
    print("Distribució de classes a y_train:")
    print(Counter(y_train))
    print("Distribució de classes a y_test:")
    print(Counter(y_test))

    for model, title in models:
        print(f"\nModel: {title}")
        model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats)


    # Guarda els resultats al fitxer JSON
    guardar_resultats_a_json(resultats)