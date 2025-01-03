import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd 
import seaborn as sns


def processament(data, labels, img_size=(128,128)):
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    features, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
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

if __name__ == "__main__":
    base_dir = "ACproject-14-grup/datasets/Data1/images_original"

    data, labels = [], []
    print("[INFO] Processant dades i extraient característiques d'audio...")
    processament(data, labels, img_size=(128,128))
    
    print("[INFO] Codificant etiquetes i normalitzant dades...")
    data = pd.DataFrame(data)
    data["label"] = labels

    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    print("\n[INFO] Dividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    print("\n[INFO] Crear i entrenar el model SVC amb hiperparàmetres especificats...")    
    models_top = [(SVC(C=5, gamma="scale", kernel="rbf", probability=True, random_state=42), "Support Vector Machine")]

    print("\n[INFO] Crear matriu confusió...")    
    for model, name in models_top:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        confusion_matr = confusion_matrix(y_test, preds, normalize='true') # Normalize the confusion matrix
        plt.figure(figsize = (16, 9))
        sns.heatmap(confusion_matr, cmap="Blues", annot=True, fmt=".2%", 
                    xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                    yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);
        plt.savefig(".png")
        plt.show()