import os
import pandas as pd
import joblib

def save_model(model, file_path):
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)

def save_preprocessor(preprocessor, file_path):
    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(preprocessor, file_path)

def load_preprocessor(file_path):
    return joblib.load(file_path)

def load_data(file_path):
    return pd.read_csv(file_path)