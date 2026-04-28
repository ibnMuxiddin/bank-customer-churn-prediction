import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self  # Fit bosqichida hech narsa o'zgarmaydi
        
    def transform(self, X):
        # Ma'lumot nusxasini olamiz (Original DataFrame o'zgarmasligi uchun)
        X_copy = X.copy()
        
        X_copy['NumOfProducts_bins'] = pd.cut(X_copy["NumOfProducts"], bins=[0,1,2,np.inf], labels=["OneProduct", "TwoProducts", "More_Than_2_Products"]) 
        
        X_copy["Balance_bins"] = pd.cut(X_copy["Balance"], bins=[-1,0,np.inf], labels=["Zero", "Non_Zero"])
        
        return X_copy