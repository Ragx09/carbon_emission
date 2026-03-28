from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from .data_pipeline import get_preprocessing_pipeline

def get_models():
    """
    Returns a dictionary of models to satisfy the requirement: 
    'compare approach with at least two other approaches.'
    """
    preprocessor = get_preprocessing_pipeline()

    # Approach 1: Linear Regression (Week 8)
    lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])

    # Approach 2: KNN Regressor (Week 4)
    knn_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', KNeighborsRegressor(n_neighbors=5))])

    # Approach 3: SVM Regressor (Week 9)
    svm_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', SVR(kernel='rbf', C=100))])

    return {
        "Linear_Regression": lr_model,
        "KNN_Regressor": knn_model,
        "SVM_Regressor": svm_model
    }