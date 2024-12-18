# Import Libraries
import argparse
import glob
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define functions
def main(args):
    # Enable autologging with MLflow for scikit-learn models
    mlflow.sklearn.autolog()

    # Read Data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column=args.target_column)

    # Train model
    train_model(args.reg_rate, X_train, y_train, y_test)

# def get_csvs_df(path):
#     if not os.path.exists(path):
#         raise RuntimeError(f'Cannot use non-existent file provided: {path}')
#     csv_files = glob.glob(f"{path}/*.csv")
#     if not csv_files:
#         raise RuntimeError(f"No CSV files found in the provided data path: {path}")
#     return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def get_csvs_df(path):
    if os.path.isfile(path):  # Check if the path is a file
        if path.endswith('.csv'):
            return pd.read_csv(path)
        else:
            raise RuntimeError(f"The provided file is not a CSV: {path}")
    elif os.path.isdir(path):  # Check if the path is a folder
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise RuntimeError(f"No CSV files found in the provided folder: {path}")
        return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        raise RuntimeError(f"Invalid path: {path}")


def split_data(df, target_column):
    # If target_column is None, assume the last column is the target
    if target_column is None:
        target_column = df.columns[-1]  # Last column as target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(reg_rate, X_train, y_train, y_test):
    with mlflow.start_run():
        model = LogisticRegression(C=1/reg_rate, solver="liblinear")
        model.fit(X_train, y_train)

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest="training_data", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01, help="Regularization rate for logistic regression")
    parser.add_argument("--target_column", dest="target_column", type=str, default=None, help="Name of the target column (if not provided, the last column is used)")

    # Parse args
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)
    
    # Parse args
    args = parse_args()

    # Run main function
    main(args)
    
    # Add space in logs
    print("\n\n")
    print("*" * 60)
