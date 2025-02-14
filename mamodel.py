from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from metrics import main
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor



with open("output1.csv") as f:
    df = pd.read_csv(f, sep=",")
df.columns = ['cpm',"start","end","process","size","avg_V","avg_A","avg_CPM"]
df.drop(columns=['start'], inplace=True)

with open("validate_answers.tsv") as f:
    y_d = pd.read_csv(f, sep="\t")

y = y_d["at_least_one"]
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
"""
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
"""
train_pool = Pool(x_train, y_train)
test_pool = Pool(x_test, y_test)

param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1],
}



#Сделать GridSearchPar
model = CatBoostRegressor(loss_function='RMSE',
                          eval_metric='MAE')

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(x_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
accuracy = mean_squared_error(y_test, y_pred)
print(f'Ошибка модели с оптимальными параметрами: {accuracy:.4f}')


vectro = pd.DataFrame({"at_least_one" : y_pred, "at_least_two" : [np.nan] * len(y_pred), "at_least_three" : [np.nan] * len(y_pred)}) 

vectro.at[0, 'at_least_two'] = max(vectro.at[0, 'at_least_one'] * 0.7791414 -0.02475854,  0.001)  
vectro.at[0, 'at_least_three'] = max(vectro.at[0, 'at_least_two'] * 0.83223596 - 0.00769715,  0.001) 

# Цикл для заполнения остальных строк
for i in range(1, len(vectro)):
    vectro.at[i, 'at_least_two'] = max( 0.001,vectro.at[i-1, 'at_least_one'] * 0.7791414 -0.02475854)
    vectro.at[i, 'at_least_three'] = max(vectro.at[i-1, 'at_least_two'] * 0.83223596 - 0.00769715, 0.001)

# Проверяем результат

def load_answers(answers_filename):
    return pd.read_csv(answers_filename, sep="\t")


def get_smoothed_log_mape_column_value(responses_column, answers_column, epsilon):
    return np.abs(np.log(
        (responses_column + epsilon)
        / (answers_column + epsilon)
    )).mean()


def get_smoothed_mean_log_accuracy_ratio(answers, responses, epsilon=0.005):
    log_accuracy_ratio_mean = np.array(
        [
            get_smoothed_log_mape_column_value(responses.at_least_one, answers.at_least_one, epsilon),
            get_smoothed_log_mape_column_value(responses.at_least_two, answers.at_least_two, epsilon),
            get_smoothed_log_mape_column_value(responses.at_least_three, answers.at_least_three, epsilon),
        ]
    ).mean()

    percentage_error = 100 * (np.exp(log_accuracy_ratio_mean) - 1)

    return percentage_error.round(
        decimals=2
    )


def main(answers, responses):

    print(get_smoothed_mean_log_accuracy_ratio(answers, responses))



main(y_d, vectro)



# Загуглить
# UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan nan nan nan nan]

