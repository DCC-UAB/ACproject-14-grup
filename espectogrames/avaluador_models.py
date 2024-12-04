"""
MODELS a implementar:

"""

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
...

# Funció on s'incorpora els models
...
