import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data  # Абсолютный импорт
import streamlit as st  # Импортируем streamlit

def visualize_basic(train_data):
    st.title("Базовая визуализация данных")

    # Описание распределения цен на дома
    st.subheader("Распределение цен на дома")
    st.markdown("""
    Гистограмма показывает распределение цен на дома. Это помогает понять, как распределены цены и есть ли выбросы.
    """)
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['SalePrice'], bins=50, kde=True)
    plt.title('Распределение цен по домам')
    plt.xlabel('Цена')
    plt.ylabel('Частота')
    st.pyplot(plt.gcf())
    plt.clf()

    # Описание гистограмм важных числовых признаков
    numeric_features = ['GrLivArea', 'YearBuilt', 'OverallQual']
    st.subheader("Гистограммы важных числовых признаков")
    st.markdown("""
    - **GrLivArea:** Жилая площадь дома. Этот признак обычно сильно коррелирует с ценой.
    - **YearBuilt:** Год постройки дома. Этот признак может быть важным для определения цены.
    - **OverallQual:** Общий уровень качества дома. Этот признак является одним из самых важных для предсказания цены.
    """)
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(1, 3, i)
        sns.histplot(train_data[feature], bins=30, kde=True)
        plt.title(f'Распределение {feature}')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Описание барплотов важных категориальных признаков
    categorical_features = ['Neighborhood', 'HouseStyle', 'MSZoning']
    st.subheader("Барплоты важных категориальных признаков")
    st.markdown("""
    Выберем несколько важных категориальных признаков, таких как Neighborhood (район), HouseStyle (стиль дома) и MSZoning (зонирование).
    - **Neighborhood:** Район, в котором находится дом. Разные районы могут иметь разные уровни цен.
    - **HouseStyle:** Стиль дома. Разные стили могут иметь разные цены.
    - **MSZoning:** Зонирование. Разные зоны могут иметь разные ограничения и, соответственно, разные цены.
    """)
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(1, 3, i)
        sns.countplot(y=feature, data=train_data)
        plt.title(feature)
        plt.xlabel('Частота')
        plt.ylabel('Категория')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Описание матрицы корреляций
    st.subheader("Матрица корреляций")
    st.markdown("""
    Матрица корреляций показывает, как признаки коррелируют друг с другом. Это помогает выявить признаки, которые могут быть полезны для модели, а также признаки, которые могут быть избыточными.
    Для вычисления матрицы корреляций нам нужно преобразовать категориальные признаки в числовые. Мы будем использовать метод get_dummies для кодирования категориальных признаков. Затем мы ограничим количество признаков для улучшения читаемости.
    """)
    # Преобразование категориальных признаков в числовые
    train_data_encoded = pd.get_dummies(train_data)
    # Выбор важных числовых признаков для матрицы корреляций
    important_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd']
    # Матрица корреляций
    plt.figure(figsize=(12, 8))
    correlation_matrix = train_data_encoded[important_features].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 10})
    plt.title('Матрица корреляций')
    plt.xlabel('Признаки')
    plt.ylabel('Признаки')
    st.pyplot(plt.gcf())
    plt.clf()