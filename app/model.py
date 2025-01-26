from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import numpy as np
from sklearn.model_selection import cross_val_predict

def train_model(data):
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict(model, data):
    predictions = model.predict(data)
    return predictions

def evaluate_model(model, data):
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return rmse

def evaluate_model_cross_validation(model, X, y):
    predictions = cross_val_predict(model, X, y, cv=5)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # Убедитесь, что значения целевой переменной и предсказаний положительны перед вычислением RMSLE
    if np.all(y >= 0) and np.all(predictions >= 0):
        rmsle = np.sqrt(mean_squared_log_error(y, predictions))
    else:
        rmsle = float('nan')  # Если есть отрицательные значения, RMSLE не может быть вычислен
    
    return rmse, mae, r2, rmsle