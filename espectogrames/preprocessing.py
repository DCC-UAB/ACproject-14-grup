import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Carrega i preprocessa imatges des del directori base.
def load_images(base_dir, img_size=(128, 128)):
    if not os.path.exists(base_dir) or not os.listdir(base_dir):
        raise FileNotFoundError(f"El directori {base_dir} és buit o no existeix.")

    data, labels = [], []
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR) # possible simplificació probar després: cv2.IMREAD_GRAYSCALE
                if img is not None:
                    img_resized = cv2.resize(img, img_size) # normalització de les imatges
                    data.append(img_resized.flatten())
                    labels.append(genre)
    
    if not data or not labels:
        raise ValueError("No s'han processat dades. Comprova el directori d'entrada.")

    return np.array(data), np.array(labels)

# Codifica les etiquetes i valida el nombre mínim de mostres per classe.
def encode_labels(labels, min_samples_per_class=2):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    label_counts = Counter(labels_encoded)
    for label, count in label_counts.items():
        if count < min_samples_per_class:
            raise ValueError(f"La classe {label} té menys de {min_samples_per_class} mostres.")

    return labels_encoded, label_encoder

# Divideix el dataset en conjunts d'entrenament i test.
def split_dataset(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42, stratify=labels)
    return X_train, X_test, y_train, y_test

# Preprocessa imatges i les divideix en conjunts d'entrenament i test.
def preprocess_images(base_dir, img_size=(128, 128), test_size=0.2, min_samples_per_class=2):
    data, labels = load_images(base_dir, img_size)
    labels_encoded, label_encoder = encode_labels(labels, min_samples_per_class)
    X_train, X_test, y_train, y_test = split_dataset(data, labels_encoded, test_size)
    return X_train, X_test, y_train, y_test, label_encoder

base_dir = "ACproject-14-grup/datasets/Data1/images_original"
X_train, X_test, y_train, y_test, label_encoder = preprocess_images(base_dir)

print(f"Imatges d'entrenament: {X_train.shape}, Imatges de test: {X_test.shape}")
print(f"Classes: {label_encoder.classes_}")
