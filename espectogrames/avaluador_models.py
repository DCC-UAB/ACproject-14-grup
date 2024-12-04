"""
MODELS a implementar:
KNN, DecisionTree, KMeans, SVM, 
RandomForest, GradientBoosting, XGBoost (XGB, XGBRF)
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



