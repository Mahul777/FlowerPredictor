# src\model_training4.py
from src.logger1 import get_logger  
from src.custom_exception2 import CustomException
from src.data_processing3 import DataProcessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns   
import os

# initializing the logger
logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path  # csv file path
        self.data = None
        # storing the data in a processed folder
        self.processed_data_path = "artifacts/processed"
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        self.model_path = "artifacts/models"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model = RandomForestClassifier(random_state=42)  # Initialize model
        logger.info("Model initialized successfully")
    
    # Load the data 
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in loading data", e)
    
    # Load the processed data
    def load_processed_data(self):
        try:
            X_train = joblib.load(f"{self.processed_data_path}/X_train_scaled.pkl")
            X_test = joblib.load(f"{self.processed_data_path}/X_test_scaled.pkl")
            y_train = joblib.load(f"{self.processed_data_path}/y_train.pkl")
            y_test = joblib.load(f"{self.processed_data_path}/y_test.pkl")
            logger.info("Processed data loaded successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in loading processed data", e)
    
    # Load the model if exists, otherwise initialize a new model
    def load_model(self):
        try:
            model_file = f"{self.model_path}/model.pkl"
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)  # Load the pre-trained model
                logger.info("Model loaded successfully")
            else:
                logger.info("Model file not found, will train a new model.")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in loading model", e)
    
    # Train the model
    def train_model(self, X_train, X_test, y_train, y_test):
        try:
            self.model.fit(X_train, y_train)
            # Save the trained model
            joblib.dump(self.model, f"{self.model_path}/model.pkl")
            logger.info("Model trained and saved successfully")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in training model", e)
    
    # Evaluate the model
    def evaluate_model(self, X_test, y_test):
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Print the evaluation metrics for each model
            print("\nModel Evaluation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            
            logger.info("Model evaluation metrics calculated successfully")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info("Confusion Matrix:")
            logger.info(conf_matrix)
            return accuracy, precision, recall, f1, conf_matrix
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in evaluating model", e)
    
    # Plot the confusion matrix
    def plot_confusion_matrix(self, conf_matrix):
        try:
            # Create a figure and axes objects
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot the confusion matrix
            sns.heatmap(conf_matrix, annot=True, ax=ax)
            
            # Set the title and axis labels
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            
            # Save the plot to a file
            plt.savefig(f"{self.model_path}/confusion_matrix.png")
            logger.info("Confusion matrix plotted successfully")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in plotting confusion matrix", e)
    
    # Run the entire model training pipeline
    def run(self):
        try:
            self.load_data()
            X_train, X_test, y_train, y_test = self.load_processed_data()
            self.load_model()  # Load model if available
            self.train_model(X_train, X_test, y_train, y_test)  # Train and save the model if not available
            accuracy, precision, recall, f1, conf_matrix = self.evaluate_model(X_test, y_test)
            self.plot_confusion_matrix(conf_matrix)
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in running the model training", e)

if __name__ == "__main__":
    model_training = ModelTraining("artifacts/raw/data.csv")
    model_training.run()
