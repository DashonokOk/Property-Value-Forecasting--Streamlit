import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sklearn

def preprocess_data(data):
    # Обработка пропущенных значений
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=[object]).columns.tolist()

    # Определение трансформеров для числовых и категориальных столбцов
    numerical_transformer = SimpleImputer(strategy='mean')
    
    # Проверка версии scikit-learn для правильной обработки параметра 'sparse'
    if sklearn.__version__ >= '1.2':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Для новых версий
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Для старых версий
        ])

    # Создание препроцессора
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Сохранение любых оставшихся столбцов (если есть)
    )

    # Подгонка и преобразование данных
    data_processed_array = preprocessor.fit_transform(data)

    # Получение имен признаков после one-hot кодирования
    num_feature_names = numerical_cols
    if sklearn.__version__ >= '1.0':
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    else:
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_cols)
        
    feature_names = num_feature_names + list(cat_feature_names)

    # Преобразование массива в DataFrame
    data_processed = pd.DataFrame(data_processed_array, columns=feature_names)

    return preprocessor, data_processed