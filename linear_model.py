from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from metrics import main


with open("validate_answers.tsv") as f:
    answr = pd.read_csv(f, sep="\t")


x = pd.DataFrame(answr["at_least_two"])
y = pd.DataFrame(answr["at_least_three"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Оцениваем качество модели
mse = np.mean((y_test - y_pred)**2)
r2 = model.score(x_test, y_test)

print("Средняя квадратичная ошибка (MSE):", mse)
print("Коэффициент детерминации (R^2):", r2)

print("Коэффициент x:", model.coef_[0])  # Наклон
print("Смещение b:", model.intercept_)  # Смещение