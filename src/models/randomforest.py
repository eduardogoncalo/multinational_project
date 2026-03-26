from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.models.base import BaseModel
from src.data_preparation.pipeline import create_model_pipeline
from typing import Optional

class RandomForest(BaseModel):
    def __init__(self, model_name: str, base_preprocessor: Optional[Pipeline] = None) -> None:
        
        super().__init__(model_name, base_preprocessor)
        
        
        self.rf_estimator = RandomForestClassifier(n_estimators=100)
        
        
        if self.base_preprocessor is not None:
            self.mdl = create_model_pipeline(self.base_preprocessor, self.rf_estimator)
        else:
            self.mdl = self.rf_estimator

    def train(self, data) -> None:
        self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def fit_transform(self, data) -> None:
        self.mdl.fit_transform(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test):
        self.predictions = self.mdl.predict(X_test)
        return self.predictions

