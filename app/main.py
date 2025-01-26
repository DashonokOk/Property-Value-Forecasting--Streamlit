import os
import sys
import streamlit as st
import pandas as pd
from utils import save_model, load_model, save_preprocessor, load_preprocessor, load_data  # Абсолютный импорт
from model import train_model, predict, evaluate_model_cross_validation  # Абсолютный импорт
from data_processing import preprocess_data  # Абсолютный импорт
from visualization import visualize_basic  # Абсолютный импорт
import matplotlib.pyplot as plt  # Импортируем plt для графиков
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.model_selection import cross_val_predict

# Установите рабочую директорию в папку 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Создание директории models, если она не существует
if not os.path.exists('models'):
    os.makedirs('models')

# Загрузка данных
train_data = load_data("data/train.csv")

# Оформление боковой панели
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Прогнозирование стоимости недвижимости", "Базовая визуализация"),
    key='navigation_radio_sidebar'  # Уникальный ключ для радиокнопки в боковой панели
)

# Страница для прогнозирования стоимости недвижимости
def prediction_page():
    st.title("Прогнозирование стоимости недвижимости")
    st.write("Загрузите файлы с данными для обучения и тестирования.")

    uploaded_train_file = st.file_uploader("Загрузите файл с данными для обучения (train.csv)", type="csv")
    uploaded_test_file = st.file_uploader("Загрузите файл с данными для тестирования (test.csv)", type="csv")

    if uploaded_train_file is not None and uploaded_test_file is not None:
        train_data = pd.read_csv(uploaded_train_file)
        test_data = pd.read_csv(uploaded_test_file)

        # Разделение признаков и целевой переменной для данных обучения
        X_train = train_data.drop(columns=["SalePrice"])
        y_train = train_data["SalePrice"]

        # Предобработка данных
        preprocessor, X_train_processed = preprocess_data(X_train)

        # Обучение модели
        model = train_model(pd.concat([X_train_processed, y_train], axis=1))
        save_model(model, "models/trained_model.pkl")
        save_preprocessor(preprocessor, "models/preprocessor.pkl")

        # Загрузка обученной модели и препроцессора
        model = load_model("models/trained_model.pkl")
        preprocessor = load_preprocessor("models/preprocessor.pkl")

        # Предобработка тестовых данных с использованием того же препроцессора
        X_test = test_data.drop(columns=["Id"])  

        # Убедитесь, что все колонки, используемые в препроцессинге, присутствуют в тестовом наборе данных
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0  # или np.nan, если вы предпочитаете заполнить позже

        # Переупорядочение колонок для соответствия данным обучения
        X_test = X_test[X_train.columns]

        # Применение препроцессора к тестовым данным
        X_test_processed = pd.DataFrame(
            preprocessor.transform(X_test),
            columns=preprocessor.get_feature_names_out()
        )

        # Убедитесь, что обработанные тестовые данные имеют те же колонки, что и обработанные данные обучения
        X_train_processed_columns = X_train_processed.columns.tolist()
        X_test_processed = X_test_processed.reindex(columns=X_train_processed_columns, fill_value=0)

        # Выполнение предсказаний на тестовых данных
        predictions = predict(model, X_test_processed)

        # Создание DataFrame для отправки
        submission = pd.DataFrame({
            'Id': test_data['Id'],
            'SalePrice': predictions
        })

        # Сохранение файла отправки в CSV
        submission.to_csv('submission.csv', index=False)

        # Отображение сообщения о том, что файл отправки создан
        st.success("Файл отправки 'submission.csv' успешно создан!")

        # Показать первые несколько строк файла отправки
        st.write("Вот первые несколько строк файла отправки:")
        st.write(submission.head())

        # Предоставление ссылки для скачивания файла отправки
        st.download_button(
            label="Скачать файл отправки",
            data=submission.to_csv(index=False),
            file_name='submission.csv',
            mime='text/csv'
        )

        # Добавляем раздел с отчетом о проделанной работе
        st.header("Отчет о проделанной работе")

        st.subheader("1. Фича-инжиниринг и предобработка данных")
        st.markdown("""
        В процессе работы над проектом мы выполнили следующие шаги по фича-инжинирингу и предобработке данных:
        - Обработка пропущенных значений с помощью `SimpleImputer`.
        - Преобразование категориальных признаков с помощью `OneHotEncoder`.
        - Исключение ненужных столбцов, которые не влияют на целевую переменную.
        """)

        st.subheader("2. Препроцессингный пайплайн")
        st.markdown("""
        Окончательный препроцессингный пайплайн включает в себя следующие шаги:
        - Заполнение пропущенных значений числовых признаков средним значением.
        - Заполнение пропущенных значений категориальных признаков наиболее частым значением.
        - Преобразование категориальных признаков в числовые с помощью One-Hot Encoding.
        """)

        st.subheader("3. Важность признаков")
        # Получаем важность признаков
        feature_importances = model.feature_importances_
        feature_names = X_train_processed.columns.tolist()

        # Создаем DataFrame для визуализации важности признаков
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Визуализируем важность признаков
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
        ax.set_xlabel('Важность')
        ax.set_title('Топ-10 важных признаков')
        st.pyplot(fig)

        st.subheader("4. Результаты кросс-валидации")
        # Оценка модели с помощью кросс-валидации
        cv_predictions = cross_val_predict(model, X_train_processed, y_train, cv=5)
        rmse = np.sqrt(mean_squared_error(y_train, cv_predictions))
        mae = mean_absolute_error(y_train, cv_predictions)
        r2 = r2_score(y_train, cv_predictions)
        
        # Убедитесь, что значения целевой переменной и предсказаний положительны перед вычислением RMSLE
        if np.all(y_train >= 0) and np.all(cv_predictions >= 0):
            rmsle = np.sqrt(mean_squared_log_error(y_train, cv_predictions))
        else:
            rmsle = float('nan')  # Если есть отрицательные значения, RMSLE не может быть вычислен

        metrics_df = pd.DataFrame({
            'Метрика': ['RMSE', 'MAE', 'R^2', 'RMSLE'],
            'Значение': [rmse, mae, r2, rmsle]
        })
        st.table(metrics_df)

        st.subheader("5. Итоговое место на Kaggle")
        st.markdown("""
        Наша модель заняла **[место]** на конкурсе Kaggle.
        """)

    else:
        st.info("Пожалуйста, загрузите обучающий и тестовый файлы.")

# Страница для визуализации данных
def visualization_page():
    st.title("Базовая визуализация данных")
    visualize_basic(train_data)

# Убедитесь, что все ключи уникальны
if page == "Прогнозирование стоимости недвижимости":
    prediction_page()
elif page == "Базовая визуализация":
    visualization_page()
else:
    st.error("Неизвестная страница.")