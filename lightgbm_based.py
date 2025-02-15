import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Загрузка данных
users = pd.read_csv('users.tsv', sep='\t')
history = pd.read_csv('history.tsv', sep='\t')
validate = pd.read_csv('validate.tsv', sep='\t')
validate_answers = pd.read_csv('validate_answers.tsv', sep='\t')

# Объединение validate и validate_answers
validate_data = validate.merge(validate_answers, left_index=True, right_index=True)

# Преобразование user_ids в список
validate_data['user_ids'] = validate_data['user_ids'].apply(lambda x: eval(x))

# Создание нового датасета для обучения
data_for_training = []

for _, row in validate_data.iterrows():
    user_ids = row['user_ids']
    cpm = row['cpm']
    hour_start = row['hour_start']
    hour_end = row['hour_end']
    publishers = row['publishers']

    # Фильтрация history для пользователей из user_ids
    filtered_history = history[history['user_id'].isin(user_ids)]

    # Агрегация данных по каждому пользователю
    aggregated_data = filtered_history.groupby('user_id').agg({
        'hour': ['count', 'min', 'max'],
        'cpm': ['mean', 'max'],
        'publisher': 'nunique'
    }).reset_index()

    # Переименование колонок
    aggregated_data.columns = [
        'user_id',
        'show_count',
        'first_show_hour',
        'last_show_hour',
        'avg_cpm',
        'max_cpm',
        'unique_publishers'
    ]

    # Добавление информации о пользователе
    aggregated_data = aggregated_data.merge(users, on='user_id', how='left')

    # Добавление признаков из validate
    aggregated_data['cpm'] = cpm
    aggregated_data['hour_start'] = hour_start
    aggregated_data['hour_end'] = hour_end
    aggregated_data['publishers'] = len(publishers)

    # Добавление целевых переменных
    aggregated_data['at_least_one'] = row['at_least_one']
    aggregated_data['at_least_two'] = row['at_least_two']
    aggregated_data['at_least_three'] = row['at_least_three']

    # Добавление данных в общий список
    data_for_training.append(aggregated_data)

# Объединение всех данных в один DataFrame
data_for_training = pd.concat(data_for_training, ignore_index=True)

# Обработка пропусков
data_for_training['age'] = data_for_training['age'].replace(0, data_for_training['age'].median())
data_for_training['city_id'] = data_for_training['city_id'].astype('category')

# Выбор признаков
features = [
    'show_count',
    'first_show_hour',
    'last_show_hour',
    'avg_cpm',
    'max_cpm',
    'unique_publishers',
    'age',
    'city_id',
    'cpm',
    'hour_start',
    'hour_end',
    'publishers'
]
X = data_for_training[features]
y_at_least_one = data_for_training['at_least_one']

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(
    X, y_at_least_one, test_size=0.2, random_state=42
)

# Создание Dataset для LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Параметры модели
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Обучение модели
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    # early_stopping_rounds=50,
    # verbose_eval=50
)

# Предсказания
y_pred = model.predict(X_val)

# Оценка качества
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error: {mae}")
print(y_pred)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# import lightgbm as lgb
#
# # Загрузка данных
# users = pd.read_csv('users.tsv', sep='\t')
# history = pd.read_csv('history.tsv', sep='\t')
# validate_data = pd.read_csv('validate.tsv', sep='\t')
# validate_answers = pd.read_csv('validate_answers.tsv', sep='\t')
#
# # Загрузка данных
# validate_data = pd.read_csv('validate.tsv', sep='\t')
# validate_answers = pd.read_csv('validate_answers.tsv', sep='\t')
#
# # Проверка наличия колонок
# print("Columns in validate_data:", validate_data.columns)
# print("Columns in validate_answers:", validate_answers.columns)
#
# # Преобразование user_ids из строки в список
# validate_data['user_ids'] = validate_data['user_ids'].apply(lambda x: list(map(int, x.split(','))))
#
# # Подготовка данных для обучения
# data_for_training = []
#
# for _, row in validate_data.iterrows():
#     user_ids = row['user_ids']
#     cpm = row['cpm']
#     hour_start = row['hour_start']
#     hour_end = row['hour_end']
#     publishers = row['publishers']
#
#     # Фильтрация history для пользователей из user_ids
#     filtered_history = history[history['user_id'].isin(user_ids)]
#
#     # Агрегация данных по каждому пользователю
#     aggregated_data = filtered_history.groupby('user_id').agg({
#         'hour': ['count', 'min', 'max'],
#         'cpm': ['mean', 'max'],
#         'publisher': 'nunique'
#     }).reset_index()
#
#     # Переименование колонок
#     aggregated_data.columns = [
#         'user_id',
#         'show_count',
#         'first_show_hour',
#         'last_show_hour',
#         'avg_cpm',
#         'max_cpm',
#         'unique_publishers'
#     ]
#
#     # Добавление информации о пользователе
#     aggregated_data = aggregated_data.merge(users, on='user_id', how='left')
#
#     # Добавление признаков из validate
#     aggregated_data['cpm'] = cpm
#     aggregated_data['hour_start'] = hour_start
#     aggregated_data['hour_end'] = hour_end
#     aggregated_data['publishers'] = len(publishers)
#
#     # Добавление целевых переменных
#     aggregated_data['at_least_one'] = row['at_least_one']
#     aggregated_data['at_least_two'] = row['at_least_two']
#     aggregated_data['at_least_three'] = row['at_least_three']
#
#     # Добавление данных в общий список
#     data_for_training.append(aggregated_data)
#
# # Объединение всех данных в один DataFrame
# data_for_training = pd.concat(data_for_training, ignore_index=True)
#
# # Обработка пропусков
# data_for_training['age'] = data_for_training['age'].replace(0, data_for_training['age'].median())
# data_for_training['city_id'] = data_for_training['city_id'].astype('category')
#
# # Выбор признаков
# features = [
#     'show_count',
#     'first_show_hour',
#     'last_show_hour',
#     'avg_cpm',
#     'max_cpm',
#     'unique_publishers',
#     'age',
#     'city_id',
#     'cpm',
#     'hour_start',
#     'hour_end',
#     'publishers'
# ]
# X = data_for_training[features]
#
# # Целевые переменные
# y_at_least_one = data_for_training['at_least_one']
# y_at_least_two = data_for_training['at_least_two']
# y_at_least_three = data_for_training['at_least_three']
#
# # Разделение данных
# X_train, X_val, y_train_one, y_val_one = train_test_split(X, y_at_least_one, test_size=0.2, random_state=42)
# _, _, y_train_two, y_val_two = train_test_split(X, y_at_least_two, test_size=0.2, random_state=42)
# _, _, y_train_three, y_val_three = train_test_split(X, y_at_least_three, test_size=0.2, random_state=42)
#
# # Создание Dataset для LightGBM
# train_data_one = lgb.Dataset(X_train, label=y_train_one)
# val_data_one = lgb.Dataset(X_val, label=y_val_one, reference=train_data_one)
#
# train_data_two = lgb.Dataset(X_train, label=y_train_two)
# val_data_two = lgb.Dataset(X_val, label=y_val_two, reference=train_data_two)
#
# train_data_three = lgb.Dataset(X_train, label=y_train_three)
# val_data_three = lgb.Dataset(X_val, label=y_val_three, reference=train_data_three)
#
# # Параметры модели
# params = {
#     'objective': 'regression',
#     'metric': 'mae',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'max_depth': -1,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': -1
# }
#
# # Обучение модели для at_least_one
# model_one = lgb.train(
#     params,
#     train_data_one,
#     valid_sets=[train_data_one, val_data_one],
#     num_boost_round=1000,
#     # early_stopping_rounds=50,
#     # verbose_eval=50
# )
#
# # Обучение модели для at_least_two
# model_two = lgb.train(
#     params,
#     train_data_two,
#     valid_sets=[train_data_two, val_data_two],
#     num_boost_round=1000,
#     # early_stopping_rounds=50,
#     # verbose_eval=50
# )
#
# # Обучение модели для at_least_three
# model_three = lgb.train(
#     params,
#     train_data_three,
#     valid_sets=[train_data_three, val_data_three],
#     num_boost_round=1000,
#     # early_stopping_rounds=50,
#     # verbose_eval=50
# )
#
# # Предсказания
# y_pred_one = model_one.predict(X_val)
# y_pred_two = model_two.predict(X_val)
# y_pred_three = model_three.predict(X_val)
#
#
# # Оценка качества
# def smoothed_mean_log_accuracy_ratio(y_true, y_pred, epsilon=0.005):
#     log_ratio = np.abs(np.log((y_pred + epsilon) / (y_true + epsilon)))
#     return 100 * (np.exp(log_ratio.mean()) - 1)
#
#
# # Вычисление метрики для всех трех показателей
# metric_one = smoothed_mean_log_accuracy_ratio(y_val_one, y_pred_one)
# metric_two = smoothed_mean_log_accuracy_ratio(y_val_two, y_pred_two)
# metric_three = smoothed_mean_log_accuracy_ratio(y_val_three, y_pred_three)
#
# print(f"Metric for at_least_one: {metric_one}")
# print(f"Metric for at_least_two: {metric_two}")
# print(f"Metric for at_least_three: {metric_three}")