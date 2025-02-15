import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open("../data/validate.tsv") as f:
    val = pd.read_csv(f, sep="\t")
with open("../data/users.tsv") as f:
    user_d = pd.read_csv(f, sep="\t")
with open("../data/history.tsv") as f:
    history = pd.read_csv(f, sep="\t")

#логарифмирование
min_non_zero = history['cpm'][history['cpm'] > 0].min()

# Заменяем 0 на минимальное положительное значение
history['cpm'] = history['cpm'].replace(0, min_non_zero)

# 2. Логарифмирование данных:
# Используем np.log() для натурального логарифма (основание e)
history['cpm_log'] = np.log(history['cpm'])

# Или можно использовать np.log1p() если есть нули и
# вы хотите избежать их предварительной замены (log(x+1))
# user_d['cpm_log'] = np.log1p(user_d['cpm'])

# Теперь столбец 'cpm_log' содержит логарифмированные значения 'cpm'

# 3. Анализ корреляции с логарифмированными данными:
correlation = history['cpm_log'].corr(history['hour'])
correlation1 = history['cpm'].corr(history['hour'])
print(f"Корреляция между log(cpm) и hour: {correlation}")
print(f"Корреляция между cpm и hour: {correlation1}")


plt.figure(figsize=(10, 6)) # Установка размера фигуры
plt.scatter(history["cpm_log"], history["hour"]) # Построение графика рассеяния
plt.title('График рассеяния')
plt.xlabel('cpm')
plt.ylabel('hour')
plt.grid(True)
plt.show()

# 5. Пересчет корреляционной матрицы:
correlation_matrix = user_d.corr()
print("\nКорреляционная матрица после логарифмирования cpm:\n", correlation_matrix)
# print(history["hour"].unique())

