from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_preparation.preprocessor.text_concatenator import TextConcatenator
from config import Config

def create_feature_pipeline(text_columns) -> ColumnTransformer:
    text_pipeline = Pipeline(steps=[
        ('concat_columns', TextConcatenator()),
        ('vectorize_text', TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text_processing', text_pipeline, text_columns )
        ],
        remainder='drop' # model training only with embeddings 
    )
    
    return preprocessor

def create_model_pipeline(preprocessor: Pipeline, model) -> Pipeline:
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', model)           
    ])
