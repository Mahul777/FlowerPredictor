# src\training_pipeline5.py
from src.data_processing3 import DataProcessing
from src.model_training4 import ModelTraining



if __name__ == "__main__":  
    data_processing = DataProcessing("artifacts/raw/data.csv")
    data_processing.run()

    model_training = ModelTraining("artifacts/raw/data.csv")
    model_training.run()