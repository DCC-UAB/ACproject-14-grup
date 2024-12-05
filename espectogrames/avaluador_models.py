
from preprocessing import preprocess_images
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
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

# (enrecordar-se després d'aplicar grid search!!!!) també el predict probabilitat!!!

# Entrenament model
def train(model, X_train, y_train):
    # Temps que triga en entrenar-se
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    return model, train_time

# Predicting model
def test(model, X_test):
    # després aplicar probabilitats!!! i afegir al evaluate logloss i roc
    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test
    return y_pred, test_time

# Avaluant model
def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "conf_matrix": conf_matrix}

# Generació de plots i matriu de confusió
def plot_evaluate(metrics, model_name, label_encoder):
    plt.figure(figsize=(10, 6))
    plt.bar(["Accuracy", "Precision", "Recall", "F1 Score"], [round(metrics["accuracy"], 4), round(metrics["precision"], 4), 
            round(metrics["recall"], 4), round(metrics["f1_score"], 4)], color=["blue", "purple", "green", "orange"])
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
                cmap="viridis",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Funció final on s'incorpora els models per avaluar
def models_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    resultats = {}
    models = [(BernoulliNB(), "Naive Bayes (BernoulliNB)"),
            (GaussianNB(), "Naive Bayes (GaussianNB)"),
            (MultinomialNB(), "Naive Bayes (MultinomialNB)"),
            (LogisticRegression(max_iter=500, random_state=42), "Logistic Regression"),
            (KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors"),
            (DecisionTreeClassifier(random_state=42), "Decision Tree"),
            #(SVC(kernel="rbf", probability=True, random_state=42), "Support Vector Machine (SVM)"), va fatal triga la vida
            (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
            #(GradientBoostingClassifier(random_state=42), "Gradient Boosting"),
            (XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), "XGBoost (XGB)"),
            (XGBRFClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), "XGBoost (XGBRF)")]
   
    for model, nom_model in models:
        print(f"\n-----Avaluant model: {nom_model}-----")
       
        try:
            model, train_time = train(model, X_train, y_train) # Entrenament
            y_pred, test_time = test(model, X_test) # Test

            #Avaluació
            metrics = evaluate(y_test, y_pred)
            metrics["train_time"] = train_time
            metrics["test_time"] = test_time

            resultats[nom_model] = metrics
            plot_evaluate(metrics, nom_model, label_encoder)

        except Exception as e:
            print(f"\n-----Error en {nom_model}: {e}-----")
            resultats[nom_model] = {"ERROR": str(e)}

        print(f"\n-----{nom_model} finalitzat!!!-----")

    return resultats


# Execució
base_dir = "ACproject-14-grup/datasets/Data1/images_original"
X_train, X_test, y_train, y_test, label_encoder = preprocess_images(base_dir)
resultats = models_evaluate(X_train, X_test, y_train, y_test, label_encoder)

# Analitzar els resultats
for model_name, metrics in resultats.items():
    print(f"\nResultats de {model_name}:")
    if "error" in metrics:
        print(f"  Error: {metrics['error']}")
    else:
        print(f"  Accuracy: {metrics['accuracy']:.2f}")
        print(f"  Training Time: {metrics['train_time']:.2f} seconds")
        print(f"  Prediction Time: {metrics['test_time']:.2f} seconds")




