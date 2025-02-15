import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open("../data/validate.tsv") as f:
    val = pd.read_csv(f, sep="\t")
with open("../data/users.tsv") as f:
    user_d = pd.read_csv(f, sep="\t")
with open("../data/history.tsv") as f:
    history = pd.read_csv(f, sep="\t")
with (open("../data/output1.csv") as f):
    output1 = pd.read_csv(f, sep=",")

output1 = output1[output1['avg_V'] > 0]
output1 = output1[output1['process'] > 0]

#логарифмирование
# min_non_zero = output1['process'][output1['process'] > 0].min()
# print(min_non_zero)

# Заменяем 0 на минимальное положительное значение
output1['process'] = output1['process'].mask(output1['process'] < 1, 1)
output1['avg_V'] = output1['avg_V'].mask(output1['avg_V'] < 1, 1)

# print(output1['pro'].descriкаbe())

# 2. Логарифмирование данных:
# Используем np.log() для натурального логарифма (основание e)
# output1['avg_V_log'] = np.log(output1['avg_V'])
output1['process_log'] = np.log(output1['process'])
output1['avg_V_log'] = np.log(output1['avg_V'])



# print(f"Минимальное значение avg_V_log: {output1['avg_V_log'].min()}")
# Или можно использовать np.log1p() если есть нули и
# вы хотите избежать их предварительной замены (log(x+1))
# user_d['cpm_log'] = np.log1p(user_d['cpm'])

# Теперь столбец 'cpm_log' содержит логарифмированные значения 'cpm'

# 3. Анализ корреляции с логарифмированными данными:
correlation = output1['process_log'].corr(output1['avg_V_log'])
correlation1 = output1['process'].corr(output1['avg_V'])
print(f"Корреляция между log() и process(avg_V): {correlation}")
print(f"Корреляция между avg_V и process: {correlation1}")


plt.figure(figsize=(10, 6)) # Установка размера фигуры
plt.scatter(output1["avg_V_log"], output1["process_log"]) # Построение графика рассеяния
plt.title('График рассеяния')
plt.xlabel('avg_V')
plt.ylabel('process')
plt.grid(True)
plt.show()


