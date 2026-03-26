from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from src.models.base import BaseModel
from src.data_preparation.pipeline import create_model_pipeline
from typing import Optional

class RandomForestChain(BaseModel):
    def __init__(self, model_name: str, base_preprocessor: Optional[Pipeline] = None) -> None:
        super().__init__(model_name, base_preprocessor)
        
       
        self.target_encoder = OrdinalEncoder()
        
        
        base_rf = RandomForestClassifier(n_estimators=100)
        
        
        self.rf_estimator = ClassifierChain(base_rf, order=[0, 1, 2])
        
        
        if self.base_preprocessor is not None:
            self.mdl = create_model_pipeline(self.base_preprocessor, self.rf_estimator)
        else:
            self.mdl = self.rf_estimator

    def train(self, data) -> None:

        y_train_text = data.get_type_y_train()
        

        y_train_encoded = self.target_encoder.fit_transform(y_train_text)
        

        self.mdl.fit(data.get_X_train(), y_train_encoded)

    def predict(self, X_test):

        predictions_encoded = self.mdl.predict(X_test)

        self.predictions = self.target_encoder.inverse_transform(predictions_encoded)
        
        return self.predictions
