from config import Config
from src.data_preparation.preprocessor.preprocess import get_input_data, de_duplication, noise_remover, translate_to_en, save_data
from src.models.data_model import Data
from src.data_preparation.preprocessor.english_translator import EnglishTranslator

from src.models.randomforestchain import RandomForestChain
from src.models.randomforest import RandomForest
from src.data_preparation.pipeline import create_feature_pipeline
from src.evaluation.tracker import ExperimentTracker


import pandas as pd
import logging
from pathlib import Path
from config import Config


def get_smart_data() -> pd.DataFrame:
    """
    Decide whether the process is zero or reads the cachebased on the Config.FORCE_REPROCESS flag
    """
    processed_path = Path(Config.PROCESSED_CSV_PATH)


    if not processed_path.exists() or Config.FORCE_REPROCESS:
        logging.info("Inicializing preprocessor (Raw -> Cleaned)...")
        df_raw = get_input_data()
        df_cleaned = run_preprocessing_pipeline(df_raw)
        
        df_cleaned.to_csv(processed_path, index=False)
        return df_cleaned

    logging.info(f"skping preprocessor: {processed_path}")
    return pd.read_csv(processed_path)

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = de_duplication(df)
    df = noise_remover(df)
    
    

    translator = EnglishTranslator(use_case=2)

    logging.info("Translating to English. {INTERACTION_CONTENT}")
    df[Config.INTERACTION_CONTENT_TRANSLATED] = translator.transform(df[Config.INTERACTION_CONTENT])
    

    logging.info("Translating to English. {TICKET_SUMMARY}")
    df[Config.TICKET_SUMMARY_TRANSLATED] = translator.transform(df[Config.TICKET_SUMMARY])
    
    df[["y3", "y4"]] = df[["y3", "y4"]].fillna("Others")

    
    df = build_chained_labels(df)


    df[["y3", "y4"]] = df[["y3", "y4"]].fillna("Others")
    return df

def build_chained_labels(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Chained Labels (Type 2, 2+3, 2+3+4)...")
    
    df['Chain_2'] = df['y2'].astype(str) + " + " + df['y3'].astype(str)
    
    df['Chain_3'] = df['Chain_2'] + " + " + df['y4'].astype(str)
    
    return df

def main():
    logging.info("Inicializing Pipeline")
    
    df_clenead = get_smart_data()
    
    logging.info(f"Dataframe Shape, {df_clenead.shape}")


    save_data(df_clenead, Config.PROCESSED_CSV_PATH)


    data = Data(
        df=df_clenead, 
        feature_columns=Config.TEXT_FEATURES, 
        target_columns=[ 'y2', 'Chain_2', 'Chain_3'] 
    )



    logging.info(f"Number rows X_train {len(data.get_X_train())}")


    print("\n Loading Feature Pipeline")
    preprocessor = create_feature_pipeline(Config.TEXT_FEATURES)


    logging.info("\n   ===")
    rf_model = RandomForest(
        model_name="RF_Classifier", 
        base_preprocessor=preprocessor
    )
    logging.info(f"RF_Classifier Training")


    print("\n Start Training RF_CLASSIFIER")



    ##########################################################################
    ############################## MultiOutput ##############################
    ##########################################################################
    rf_model.train(data)

    logging.info(f"Model Trained")

    print("\n Predicting Test")
    X_test = data.get_X_test()
    predicoes = rf_model.predict(X_test)
    

    rf_model.train(data)
    predicoes = rf_model.predict(X_test)
    


    logging.info(f"Tracking Results MlFlow")
    
    
    tracker = ExperimentTracker(experiment_name="Multioutput_Tickets_Classification")
    
    tracker.log_experiment(
        model_name=rf_model.model_name,
        model_pipeline=rf_model.mdl, 
        data=data,
        y_test=data.get_type_y_test(),
        y_pred=predicoes
    )

    ##########################################################################
    ############################## CHAIN MODELL ##############################
    ##########################################################################
    logging.info("\n  ChainModel Started")
    rf_model_chain = RandomForestChain(
        model_name="RF_Classifier_Chain", 
        base_preprocessor=preprocessor
    )

    print("\n Start Training RF_CHAIN_CLASSIFIER")

    rf_model_chain.train(data)

    logging.info(f"Model Trained")


    print("\n Predicting Test")
    X_test = data.get_X_test()
    predicoes = rf_model_chain.predict(X_test)
    


    rf_model_chain.train(data)
    predicoes = rf_model_chain.predict(X_test)
    

    print("\n Saving Results Mlflow")
    
    
    tracker = ExperimentTracker(experiment_name="ChainClassifier_Tickets_Classification")
    
    tracker.log_experiment(
        model_name=rf_model_chain.model_name,
        model_pipeline=rf_model_chain.mdl, 
        data=data,
        y_test=data.get_type_y_test(),
        y_pred=predicoes
    )

    logging.info(f"Tracking Results Chain model MlFlow")

if __name__ == "__main__":
    main()
