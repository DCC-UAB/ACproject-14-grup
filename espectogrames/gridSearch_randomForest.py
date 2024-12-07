from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search_random_forest(X_train, y_train):

    param_grid = {"n_estimators": [100, 300, 500, 1000],  
                  "max_depth": [10, 15, 20, None],      
                  "max_features": ["sqrt", "log2"], 
                  "min_samples_split": [2, 5, 10, 15],  
                  "min_samples_leaf": [1, 2, 4, 6]     
                   }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)
    print(f"Millors paràmetres trobats: {grid_search.best_params_}")
    print(f"Millor accuracy de cross-validation: {grid_search.best_score_}")

    return grid_search.best_estimator_  
