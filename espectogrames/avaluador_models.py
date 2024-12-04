
from preprocessing import preprocess_images
base_dir = "ACproject-14-grup/datasets/Data1/images_original"
X_train, X_test, y_train, y_test, label_encoder = preprocess_images(base_dir)

# Esquema de les crides a funcions de models (NO es codi final!!)

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
# Naive Bayes
naive_bernoulli = BernoulliNB()
naive_categorical = CategoricalNB()
naive_gaussian = GaussianNB()
naive_multinomial = MultinomialNB()

from sklearn.linear_model import LogisticRegression, SGDClassifier
# Logistic Regression i SDG
logistic_regression = LogisticRegression(max_iter=500, random_state=42)
sdg = SGDClassifier(random_state=42)

from sklearn.neighbors import KNeighborsClassifier
# KNN
knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.tree import DecisionTreeClassifier
# Decission Tree
decission_tree = DecisionTreeClassifier(random_state=42)

from sklearn.cluster import KMeans
# K-means
kmeans = KMeans(n_clusters=3, random_state=42)

from sklearn.svm import SVC
# SVC
svc = SVC(kernel="rbf", probability=True, random_state=42)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Random Forest i Gradient Boosting
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

from xgboost import XGBClassifier, XGBRFClassifier
#XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_rf = XGBRFClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

# Funció d'avaluació models (enrecordar-se després d'aplicar grid search!!!!)
import time
# Entrenament model
def train(model, X_train, y_train):
    # Temps que triga en entrenar-se
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    return f"Entrenant model: {model.__class__.__name__}", model,f"Temps trigat: {train_time}"

# Predicting model
def testing(model, X_test, y_test):
    # després aplicar probabilitats!!!
    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test
    return f"Predicting model: {model.__class__.__name__}", y_pred,f"Temps trigat: {test_time}"
    

# Funció on s'incorpora els models
...
