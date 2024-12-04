"""
MODELS a implementar:
Naive Bayes (BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB), LogisticRegression, KNN, SGD, DecisionTree, KMeans, SVM, 
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


