# src\data_processing3.py

from src.logger1 import get_logger  
from src.custom_exception2 import CustomException
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# initializing the logger
logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, data_path):
        self.data_path = data_path  # csv file path
        self.data = None
        # storing the data in a processed folder
        self.processed_data_path = "artifacts/processed"
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in loading data", e)
    
    # Handle Outliers using IQR Method
    def handle_outliers(self):
        try:
            # Select only numeric columns
            numeric_data = self.data.select_dtypes(include=[np.number])
            
            # Calculate the interquartile range (IQR)
            iqr = np.subtract(*np.percentile(numeric_data, [75, 25], axis=0))
            
            # Set the threshold for outliers
            threshold = 1.5 * iqr
            
            # Create a boolean mask for outliers
            outliers = np.abs(numeric_data - np.mean(numeric_data)) > threshold
            
            # Set the outliers to NaN
            self.data[outliers] = np.nan
            logger.info("Outliers handled successfully")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in handling outliers", e)
    
    # Define features (X) and target (y)
    def define_features(self):
        try:
            # Define features (X) and target (y)
            X = self.data.drop(columns=['Species'])
            y = self.data['Species']
            logger.info("Features and target defined successfully")
            return X, y
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in defining features", e)
    
    # Split the dataset into training and testing sets and save each part in a separate .pkl file
    def split_data(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Save each split in a separate .pkl file
            joblib.dump(X_train, f"{self.processed_data_path}/X_train.pkl")
            joblib.dump(X_test, f"{self.processed_data_path}/X_test.pkl")
            joblib.dump(y_train, f"{self.processed_data_path}/y_train.pkl")
            joblib.dump(y_test, f"{self.processed_data_path}/y_test.pkl")
            
            logger.info("Split data saved successfully in separate files")
            return X_train, X_test, y_train, y_test  # Return splits for further use (like standardization)
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in splitting data", e)
    
    # Standardize the features
    def standardize_features(self, X_train, X_test):
        try:
            # Check for NaN values and handle them (e.g., fill NaNs with the mean of the column)
            X_train = X_train.fillna(X_train.mean())  # Replacing NaNs with column mean
            X_test = X_test.fillna(X_test.mean())    # Replacing NaNs with column mean
            
            # Check for constant columns (zero variance)
            X_train = X_train.loc[:, X_train.nunique() > 1]  # Drop columns with only one unique value
            X_test = X_test.loc[:, X_test.nunique() > 1]    # Drop columns with only one unique value
            
            # Standardize the features for models that require scaling (like SVM, KNN, Logistic Regression)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # Transform X_test using the same scaler
            
            logger.info("Features standardized successfully")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in standardizing features", e)
    
    # Run the entire data processing pipeline
    def run(self):
        try:
            self.load_data()
            self.handle_outliers()
            X, y = self.define_features()
            X_train, X_test, y_train, y_test = self.split_data(X, y)  # Get splits
            X_train_scaled, X_test_scaled = self.standardize_features(X_train, X_test)  # Standardize both sets
            # You can save or return the scaled datasets if needed for further processing
            joblib.dump(X_train_scaled, f"{self.processed_data_path}/X_train_scaled.pkl")
            joblib.dump(X_test_scaled, f"{self.processed_data_path}/X_test_scaled.pkl")
        except Exception as e:
            logger.error(e)
            raise CustomException("Error in running the data processing", e)

if __name__ == "__main__":
    data_processing = DataProcessing("artifacts/raw/data.csv")
    data_processing.run()


