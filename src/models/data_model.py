import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
import random

# Configuração de sementes para reprodutibilidade
seed = 0
random.seed(seed)
np.random.seed(seed)

class Data:
    def __init__(self, df: pd.DataFrame, feature_columns: list, target_columns: list) -> None:
        self.df = df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        

        valid_features = [col for col in feature_columns if col in df.columns]
        self.X = df[valid_features] 
        

        if 'y' not in df.columns:
            raise ValueError("y not in Dataframe")
            
        y_filter = df['y'] 
        Y_targets = df[target_columns]
        

        good_y_value = y_filter.value_counts()[y_filter.value_counts() >= 3].index

        if len(good_y_value) < 1:
            print("No classes with more then 3 records")
            self.X_train = None
            return


        mask = y_filter.isin(good_y_value)
        

        y_filter_good = y_filter[mask]
        Y_targets_good = Y_targets[mask] 
        X_good = self.X[mask] 
        df_good = df[mask].reset_index(drop=True)


        (self.X_train, self.X_test, 
         self.y_train, self.y_test, 
         self.train_df, self.test_df) = train_test_split(
            X_good, Y_targets_good, df_good, 
            test_size=0.2, random_state=seed, stratify=y_filter_good
        )
        
        self.y = Y_targets_good 
        self.classes = good_y_value

    def get_type(self):
        return self.y
        
    def get_X_train(self):
        return self.X_train
        
    def get_X_test(self):
        return self.X_test
        
    def get_type_y_train(self):
        return self.y_train
        
    def get_type_y_test(self):
        return self.y_test
        
    def get_train_df(self):
        return self.train_df
        
    def get_type_test_df(self):
        return self.test_df
        
