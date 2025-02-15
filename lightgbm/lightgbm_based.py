import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Загрузка данных
users = pd.read_csv('../data/users.tsv', sep='\t')
history = pd.read_csv('../data/history.tsv', sep='\t')
validate = pd.read_csv('../data/validate.tsv', sep='\t')
validate_answers = pd.read_csv('../data/validate_answers.tsv', sep='\t')

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
    filtered_history = history[history['user_id'].isin(user_ids)] # кому реально показали фильтрованное хистори

    # Агрегация данных по каждому пользователю
    aggregated_data = filtered_history.groupby('user_id').agg({
        'hour': ['count', 'min', 'max'],
        'cpm': ['mean', 'max'],
        'publisher': 'nunique' #set(все публикаторы)
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
# meow


# Логарифмируем столбцы с cpm и unique_publishers, так как они могут иметь большой разброс
data_for_training['log_avg_cpm'] = np.log1p(data_for_training['avg_cpm'])  # log1p используется для того, чтобы избежать ошибок с нулями
data_for_training['log_max_cpm'] = np.log1p(data_for_training['max_cpm'])
data_for_training['log_unique_publishers'] = np.log1p(data_for_training['unique_publishers'])
data_for_training['log_first_show_hour'] = np.log1p(data_for_training['first_show_hour'])
data_for_training['log_show_count'] = np.log1p(data_for_training['show_count'])
data_for_training['log_avg_cpm'] = np.log1p(data_for_training['avg_cpm'])
data_for_training['log_age'] = np.log1p(data_for_training['age'])
# data_for_training['log_city_id'] = np.log1p(data_for_training['city_id'])
data_for_training['log_cpm'] = np.log1p(data_for_training['cpm'])
data_for_training['log_hour_start'] = np.log1p(data_for_training['hour_start'])
data_for_training['log_hour_end'] = np.log1p(data_for_training['hour_end'])
data_for_training['log_publishers'] = np.log1p(data_for_training['publishers'])
data_for_training['log_last_show_hour'] = np.log1p(data_for_training['last_show_hour'])  # Это строка была пропущена



# Выбор признаков
features = [
    'log_show_count',
    'log_first_show_hour',
    'log_last_show_hour',
    'log_avg_cpm',
    'log_max_cpm',
    'log_unique_publishers',
    'log_age',
    'city_id',
    'log_cpm',
    'log_hour_start',
    'log_hour_end',
    'log_publishers'
]
X = data_for_training[features]

X.to_csv('lightgbm/outputSanya.csv', index=False)

#
# y_at_least_one = data_for_training['at_least_one']
# # print(validate_data)
#
# # Разделение данных
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y_at_least_one, test_size=0.2, random_state=42
# )
#
# # Создание Dataset для LightGBM
# train_data = lgb.Dataset(X_train, label=y_train)
# val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
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
# # Обучение модели
# model = lgb.train(
#     params,
#     train_data,
#     valid_sets=[train_data, val_data],
#     num_boost_round=1000,
#     # early_stopping_rounds=50,
#     # verbose_eval=50
# )
#
# # Предсказания
# y_pred = model.predict(X_val)
#
# # Оценка качества
# mae = mean_absolute_error(y_val, y_pred)
# print(f"Mean Absolute Error: {mae}")
# print(y_pred)

