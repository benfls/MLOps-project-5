import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.logger import get_logger
from src.custom_exceptiom import CustomException

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, processed_data_path = "artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"

        os.makedirs(self.model_dir, exist_ok=True)

        logger.info("Model Training Initialized.....")

    def loading_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Data loaded for model training.")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data")
        
    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(self.X_train, self.y_train)

            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))

            logger.info("Model trained and saved successfully....")

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(f"Error during model training")
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)

            y_proba = self.model.predict_proba(self.X_test)[:,1] if len(self.y_test.unique()) == 2 else None

            accuracy_score_ = accuracy_score(self.y_test, y_pred)
            precision_score_ = precision_score(self.y_test, y_pred, average='weighted')
            recall_score_ = recall_score(self.y_test, y_pred, average='weighted')
            f1_score_ = f1_score(self.y_test, y_pred, average='weighted')

            mlflow.log_metric("accuracy", accuracy_score_)
            mlflow.log_metric("precision", precision_score_)
            mlflow.log_metric("recall", recall_score_)
            mlflow.log_metric("f1_score", f1_score_)

            logger.info(f" Accuracy: {accuracy_score_} \n Precision: {precision_score_} \n Recall: {recall_score_} \n F1 Score: {f1_score_}")

            roc_auc = roc_auc_score(self.y_test, y_proba)

            mlflow.log_metric("roc_auc", roc_auc)

            logger.info(f"ROC AUC Score: {roc_auc}")
            logger.info("Model evaluation completed successfully....")

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(f"Error during model prediction")
        
    def run(self):
        try:
            self.loading_data()
            self.train_model()
            self.evaluate_model()

            logger.info("Training pipeline executed successfully....")

        except Exception as e:
            logger.error(f"Error runing the training pipeline : {e}")
            raise CustomException(f"Error running the training pipeline") 
        
if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.set_experiment("Model_Training_Experiment")
        trainer = ModelTraining()
        trainer.run()