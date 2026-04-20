import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Tangkap parameter dari mlproject
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    base_path = sys.argv[3] if len(sys.argv) > 3 else "Sleep_health_and_lifestyle_dataset_preprocessing/"

    print(f"Menjalankan model dengan n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Membaca data dari folder: {base_path}")

    # Load data
    X_train = pd.read_csv(base_path + "X_train_clean.csv")
    X_test = pd.read_csv(base_path + "X_test_clean.csv")
    y_train = pd.read_csv(base_path + "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path + "y_test.csv").squeeze()

    # menyimpan snippet atau sample input
    input_example = X_train[0:5]

    with mlflow.start_run():
        # Aktifkan autologging untuk sklearn
        mlflow.sklearn.autolog(log_models=False)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Simpan model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="Sleep_Model"
        )

        print(f"Berhasil! Akurasi: {accuracy:.4f}")
        
