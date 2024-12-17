## Import Libraries
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
    X_train, X_test, y_train, y_test = split_data(df)
    # Train model
    train_model(args.reg_rate, X_train, y_train, y_test)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f'Cannot use non-existent file provided: {path}')
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in the provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(reg_rate, X_train, y_train, y_test):
    with mlflow.start_run():
        model = LogisticRegression(C=1/reg_rate, solver="liblinear")
        model.fit(X_train, y_train)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01)
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