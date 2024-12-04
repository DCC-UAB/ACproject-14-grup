
from preprocessing import preprocess_images
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# Naive Bayes
naive_bernoulli = BernoulliNB()
naive_categorical = CategoricalNB()
naive_gaussian = GaussianNB()
naive_multinomial = MultinomialNB()

# Logistic Regression i SDG
logistic_regression = LogisticRegression(max_iter=500, random_state=42)
sdg = SGDClassifier(random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Decission Tree
decission_tree = DecisionTreeClassifier(random_state=42)

# K-means
kmeans = KMeans(n_clusters=3, random_state=42)

# SVC
svc = SVC(kernel="rbf", probability=True, random_state=42)

# Random Forest i Gradient Boosting
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

#XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_rf = XGBRFClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

# Funció d'avaluació models (enrecordar-se després d'aplicar grid search!!!!)

# Entrenament model
def train(model, X_train, y_train):
    # Temps que triga en entrenar-se
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    return f"Entrenant model: {model.__class__.__name__}", model,f"Temps trigat: {train_time}"

# Predicting model
def test(model, X_test):
    # després aplicar probabilitats!!! i afegir al evaluate logloss i roc
    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test
    return f"Predicting model: {model.__class__.__name__}", y_pred,f"Temps trigat: {test_time}"

# Avaluant model
def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)
    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "conf_matrix": conf_matrix}

# Generació de plots
def plot_evaluate(metrics, model_name):
    plt.figure(figsize=(10, 6))
    plt.bar(["Accuracy", "Precision", "Recall", "F1 Score"], [round(metrics["accuracy"], 2), round(metrics["precision"], 2), 
            round(metrics["recall"], 2), round(metrics["f1_score"], 2)], color=["blue", "purple", "green", "orange"])
    plt.title(f"{model_name} - Metrics")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metrics")
    plt.show()

    #Matriu de confusió
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics["conf_matrix"],
                annot=True,
                fmt="d",
                cmap="viridis")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Funció final on s'incorpora els models per avaluar
def models_evaluate(X_train, X_test, y_train, y_test, label_encoder):
